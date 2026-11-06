"""Parameterization of the original WorldVLA runtime; target/cache unchanged."""
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import json
import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache
from .rich import context


def load_upstream(repo):
    repo = Path(repo).expanduser().resolve()
    code = repo/'worldvla'
    if not (code/'model').is_dir() or not (code/'data/item_processor.py').is_file():
        raise ValueError('Expected WorldVLA upstream repository root (branch worldvla)')
    sys.path[:0] = [str(code), str(repo)]
    from model import ChameleonXLLMXForConditionalGeneration_ck
    from model.chameleon import ChameleonForConditionalGeneration
    return ChameleonXLLMXForConditionalGeneration_ck, ChameleonForConditionalGeneration


def make_processor(weights, device):
    from data.item_processor import FlexARItemProcessor_Action, generate_crop_size_list
    from data.convertsation import Conversation
    from xllmx.data.item_processor import MMConvItemProcessor
    import model.chameleon_vae_ori as vae

    class Processor(FlexARItemProcessor_Action):
        def __init__(self):
            MMConvItemProcessor.__init__(self, {'<|image|>': self.process_image, '<|action|>': self.process_action}, ['<|image|>', '<|action|>'], str(weights/'base_model'), Conversation)
            self.patch_size = 32
            self.crop_size_list = generate_crop_size_list(64, 32)
            self.chameleon_ori_vocab = vae.VocabInfo(json.loads((weights/'chameleon/tokenizer/text_tokenizer.json').read_text())['model']['vocab'])
            self.chameleon_ori_translation = vae.VocabTranslation(self.chameleon_ori_vocab, device=device)
            self.chameleon_ori_image_tokenizer = vae.ImageTokenizer(str(weights/'chameleon/tokenizer/vqgan.yaml'), str(weights/'chameleon/tokenizer/vqgan.ckpt'), device=device)
            self.n_bins, self.min_action, self.max_action = 256, -1, 1
            self.bins = np.linspace(-1, 1, 256)
            self.bin_centers = (self.bins[:-1]+self.bins[1:])/2
        def prompt(self, images, instruction):
            item = {'conversations': [{'from': 'human', 'value': 'What action should the robot take to '+instruction+'?'+'<|image|>'*len(images)}], 'image': images, 'action': []}
            return torch.tensor(self.process_item(item, training_mode=False), device=device, dtype=torch.long)[None]
    return Processor()


class Runtime:
    def __init__(self, suite, upstream_repo, model_root, model_path=None, device='cuda:0'):
        torch.set_num_threads(4)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Full upstream WorldVLA inference requires CUDA; use worldvla.test_cpu for CPU contracts')
        # Upstream image processor uses bare .cuda(); bind its current device
        # without rewriting upstream code or changing image-token semantics.
        torch.cuda.set_device(self.device)
        target, self.base_class = load_upstream(upstream_repo)
        root = Path(model_root).expanduser().resolve()
        self.suite = suite
        self.horizon = 10 if suite == 'libero_10' else 5
        self.processor = make_processor(root, str(self.device))
        self.model = target.from_pretrained(str(model_path or root/'model_256'/suite), torch_dtype=torch.float32, attn_implementation='sdpa', device_map=str(self.device), mask_image_logits=False).eval()
        self.model.requires_grad_(False)
        self.generation = GenerationConfig(max_new_tokens=self.horizon*12, max_length=4096, temperature=1, top_k=None, do_sample=False, eos_token_id=[8710], pad_token_id=0, use_cache=True)
        self.reset()

    def reset(self):
        self.counts = dict(branch_calls=[0, 0, 0], block_calls=0, recovery_calls=0, recovery_forward_calls=0, prefill_calls=0)
        self.times = {}

    def clock(self):
        import time
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    @torch.inference_mode()
    def native(self, prompt):
        self.model.init_input_ids = None
        out = self.model.generate(prompt, generation_config=self.generation, training=False, return_dict_in_generate=True, output_hidden_states=True)
        tokens = out.sequences[0, prompt.shape[1]:].tolist()
        return tokens, out.hidden_states[0][-1][0, -1].float().cpu()

    @torch.inference_mode()
    def prefill(self, prompt):
        self.counts['prefill_calls'] += 1
        start = self.clock()
        self.model.init_input_ids = None
        out = self.model(input_ids=prompt, past_key_values=DynamicCache(), use_cache=True, output_hidden_states=True, return_dict=True)
        self.times['prefill_s'] = self.clock()-start
        start = self.clock()
        rich = context(out, prompt, self.current_obs)
        self.times['rich_context_s'] = self.clock()-start
        return out, rich

    @torch.inference_mode()
    def block(self, prompt, proposed, prefill):
        self.counts['block_calls'] += 1
        start = self.clock()
        prefix_len = prompt.shape[1]
        draft = torch.tensor(proposed, device=self.device, dtype=torch.long)[None]
        full = torch.cat((prompt, draft), dim=1)
        mask = self.model.generate_att_mask_3(full)[:, None, prefix_len:, :]
        additive = torch.zeros(mask.shape, device=self.device, dtype=self.model.dtype).masked_fill(~mask, torch.finfo(self.model.dtype).min)
        out = self.base_class.forward(self.model, input_ids=draft, past_key_values=prefill.past_key_values, attention_mask=additive, position_ids=torch.arange(prefix_len, full.shape[1], device=self.device)[None], cache_position=torch.arange(prefix_len, full.shape[1], device=self.device), use_cache=True, return_dict=True)
        logits = torch.cat((prefill.logits[:, -1:], out.logits[:, :-1]), dim=1)
        result = logits.argmax(-1)[0].tolist(), out.past_key_values
        self.times['verify_s'] = self.clock()-start
        return result

    @torch.inference_mode()
    def recover(self, prompt, accepted, cache, max_new=None):
        """Original crop-to-one-token-behind accepted prefix and native AR suffix."""
        self.counts['recovery_calls'] += 1
        start = self.clock()
        full = torch.cat((prompt, torch.tensor(accepted, device=self.device, dtype=torch.long)[None]), dim=1)
        length = full.shape[1]
        if isinstance(cache, tuple):
            cache = DynamicCache.from_legacy_cache(cache)
        cache.crop(length-1)
        self.model.init_input_ids = full[:, :-1].clone()
        generated = []
        for _ in range(max_new or self.horizon*12-len(accepted)):
            token = full[:, -1:] if not generated else torch.tensor([[generated[-1]]], device=self.device)
            self.counts['recovery_forward_calls'] += 1
            out = self.model(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            next_token = int(out.logits[0, -1].argmax())
            generated.append(next_token)
            if next_token == 8710:
                break
        self.times['recovery_s'] = self.clock()-start
        return accepted+generated


def environment(suite, task_id, init_id):
    from libero.libero import benchmark
    from libero_util.libero_utils import get_libero_env, get_libero_dummy_action
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    env, instruction = get_libero_env(task, 256)
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(task_id)[init_id])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    env.sim.forward()
    env._update_observables(force=True)
    obs = env.env._get_observations()
    obs = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()}
    return env, instruction, obs


def image(obs):
    return Image.fromarray(obs['agentview_image'][::-1, ::-1])
