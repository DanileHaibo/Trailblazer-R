"""Official model adapters. Heavy dependencies load only for the chosen target."""
import time
from types import SimpleNamespace
import numpy as np
import torch


def state(obs):
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat'].copy()), obs['robot0_gripper_qpos']]).astype(np.float32)


def checked_action(value, N):
    value = np.asarray(value, np.float32)
    if value.shape != (N, 7) or not np.isfinite(value).all():
        raise ValueError('Nonfinite or wrong-shape action chunk')
    return value


class NoraPolicy:
    name, N, M = 'nora', 5, 2

    def __init__(self, assets, suite, config):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
        self.n = SimpleNamespace()
        self.n.fast_tokenizer = AutoProcessor.from_pretrained(assets['fast'], trust_remote_code=True)
        self.n.fast_tokenizer.time_horizon = 5
        self.n.fast_tokenizer.action_dim = 7
        path = assets[suite]
        self.n.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.n.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
        self.n.model.generation_config = GenerationConfig.from_pretrained(path)
        self.n.model.generation_config.do_sample = False
        self.limit = self.n.model.generation_config.max_new_tokens or self.n.model.generation_config.max_length
        self.bounds = np.asarray(config['nora_bounds'][suite])

    def inputs(self, obs, instruction):
        # Match original JPEG roundtrip + TF Lanczos3 exactly; not PIL approximation.
        import tensorflow as tf
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        image = obs['agentview_image'][::-1, ::-1].copy()
        image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224), method='lanczos3', antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': Image.fromarray(image), 'resized_height': 224, 'resized_width': 224}, {'type': 'text', 'text': instruction}]}]
        text = self.n.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.n.processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt')
        return {k: v.cuda() for k, v in inputs.items()}

    def encode(self, action):
        return [151665+int(t) for t in self.n.fast_tokenizer(action[None])[0]]+[151645]

    def decode(self, tokens):
        ids = [int(t)-151665 for t in tokens if 151665 <= int(t) <= 153712]
        coefficients = self.n.fast_tokenizer.bpe_tokenizer.decode(ids)
        if len(coefficients) != 35:
            raise ValueError(f'Malformed teacher chunk: {len(coefficients)} coefficients; expected 35; no zero fallback')
        return checked_action(self.n.fast_tokenizer.decode([ids], time_horizon=5, action_dim=7)[0], 5)

    def commands(self, y):
        lo, hi = self.bounds
        action = .5*(np.asarray(y)+1)*(hi-lo)+lo
        action[..., -1] = -np.sign(2*action[..., -1]-1)
        action[..., -1] = np.where(action[..., -1] >= 0, 1., action[..., -1])
        # Historical NORA mapping uses float64 bounds; preserve the command dtype.
        if action.shape != (5, 7) or not np.isfinite(action).all():
            raise ValueError('Invalid NORA commands')
        return action

    @torch.inference_mode()
    def query(self, obs, instruction):
        torch.cuda.synchronize(); start = time.perf_counter()
        inputs = self.inputs(obs, instruction)
        self.n.model.rope_deltas = None
        values = self.n.model.generate(**inputs)
        tokens = values[0, inputs['input_ids'].shape[1]:].tolist()
        y = self.decode(tokens); commands = self.commands(y)
        torch.cuda.synchronize()
        return commands, y, dict(tokens=tokens, policy_ms=(time.perf_counter()-start)*1000)


class PiPolicy:
    name, N, M = 'pi0', 10, 5

    def __init__(self, assets):
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        from safetensors.torch import load_file
        from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly
        if not check_whether_transformers_replace_is_installed_correctly():
            raise RuntimeError('Install the pinned official LeRobot transformers fork')
        path = assets['checkpoint']
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.action_tokenizer_name = assets['tokenizer']; cfg.text_tokenizer_name = assets['text_tokenizer']
        cfg.temperature = 0.; cfg.device = 'cuda'; cfg.gradient_checkpointing = False
        if cfg.chunk_size != 10 or cfg.n_action_steps != 10:
            raise ValueError('Expected released H10 policy; execute prefix 5 externally')
        self.p = PI0FastPolicy.from_pretrained(path, config=cfg, strict=True).cuda().eval().requires_grad_(False)
        saved = self.p._fix_pytorch_state_dict_keys(load_file(path+'/model.safetensors'), cfg)
        current = self.p.state_dict()
        for key, value in saved.items():
            key = key if key.startswith('model.') else 'model.'+key
            if key not in current or not torch.equal(current[key].cpu(), value.to(current[key].dtype)):
                raise ValueError('Official checkpoint tensor verification failed: '+key)
        self.pre, self.post = make_pre_post_processors(cfg, path, preprocessor_overrides={'device_processor': {'device': 'cuda'}, 'tokenizer_processor': {'tokenizer_name': cfg.text_tokenizer_name}, 'action_tokenizer_processor': {'action_tokenizer_name': assets['tokenizer'], 'paligemma_tokenizer_name': cfg.text_tokenizer_name}})
        original = self.p.decode_actions_with_fast
        def checked_decode(ids, *, time_horizon, action_dim):
            from scipy.fftpack import idct
            if (time_horizon, action_dim) != (10, 7):
                raise ValueError('Wrong PI0 FAST horizon')
            expected = []
            for token in ids:
                coeff = np.array(list(map(ord, self.p.action_tokenizer.bpe_tokenizer.decode(token)))) + self.p.action_tokenizer.min_token
                coeff = np.pad(coeff[:70], (0, max(0, 70-len(coeff)))).reshape(10, 7)
                expected.append(idct(coeff/self.p.action_tokenizer.scale, axis=0, norm='ortho'))
            decoded = original(ids, time_horizon=time_horizon, action_dim=action_dim)
            if not np.allclose(decoded, np.stack(expected), rtol=1e-6, atol=1e-6):
                raise ValueError('PI0 release decoder exception fallback detected')
            return decoded
        self.p.decode_actions_with_fast = checked_decode

    def encode(self, action):
        tok = self.p._paligemma_tokenizer
        fast = self.p.action_tokenizer(action[None])[0]
        return tok.encode('Action: ', add_special_tokens=False)+[int(tok.vocab_size-1-self.p.config.fast_skip_tokens-t) for t in fast]+tok.encode('|', add_special_tokens=False)+[tok.eos_token_id]

    def decode(self, tokens):
        return checked_action(self.p.detokenize_actions(torch.tensor([tokens], device='cuda'), 10, 7)[0].float().cpu().numpy(), 10)

    @torch.inference_mode()
    def query(self, obs, instruction):
        torch.cuda.synchronize(); start = time.perf_counter()
        batch = {'observation.state': torch.from_numpy(state(obs)), 'task': instruction}
        for key, raw in [('image', 'agentview_image'), ('image2', 'robot0_eye_in_hand_image')]:
            batch['observation.images.'+key] = torch.from_numpy(obs[raw][::-1, ::-1].copy()).permute(2, 0, 1).float()/255
        captured = []
        original = self.p.model.sample_actions_fast_kv_cache
        def sampler(*args, **kwargs):
            value = original(*args, **kwargs)
            captured.append(value[0].tolist())
            return value
        self.p.model.sample_actions_fast_kv_cache = sampler
        try:
            normalized = self.p.predict_action_chunk(self.pre(batch))
        finally:
            self.p.model.sample_actions_fast_kv_cache = original
        commands = self.post(normalized)[0].cpu().numpy()
        y = checked_action(normalized[0].float().cpu().numpy(), 10)
        checked_action(commands, 10)
        torch.cuda.synchronize()
        return commands, y, dict(tokens=captured[0], policy_ms=(time.perf_counter()-start)*1000)
