"""Parameterized original context_v4 + transformer_runtime_v1 + runtime recovery.
One target prefill, three independent fresh drafts, one official masked block.
The verifier is candidate-prefix-conditioned, not an independent AR oracle.
"""
from pathlib import Path
import torch
from transformers import GenerationConfig
from transformers.cache_utils import DynamicCache
from .draft import TransformerDraft, BRANCHES, context_feature
from .gate import AbsoluteGate

class Runtime:
    def __init__(self, suite, *, base_path, processor, checkpoint_dir=None, gate_config=None, mode='spec', device='cuda'):
        from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
        self.suite, self.processor, self.device = suite, processor, device
        base = Path(base_path)
        if not (base/'config.json').exists():
            base = base/suite
        self.model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            str(base), torch_dtype=torch.bfloat16, device_map=device,
            mask_image_logits=False, max_position_embeddings=8192).eval()
        self.model.requires_grad_(False)
        self.horizon = int(self.model.config.time_horizon)
        # Match official discrete generate_dis_ma budget (300), not just horizon*12.
        self.config = GenerationConfig(max_new_tokens=300, max_length=8192,
            do_sample=False, temperature=1, top_k=None, eos_token_id=[8710])
        self.max_new_tokens = 300
        self.drafts = []
        if mode == 'spec':
            if gate_config is None or checkpoint_dir is None:
                raise ValueError('Spec requires explicit gate and checkpoint directory')
            self.absolute_gate = AbsoluteGate(gate_config)
            for name, lo, hi in BRANCHES:
                ck = torch.load(Path(checkpoint_dir)/(suite+'_'+name+'_best.pt'), map_location='cpu', weights_only=False)
                if ck['horizon'] != self.horizon or ck['dims'] != hi-lo:
                    raise ValueError('Checkpoint chunk/dimension mismatch')
                draft = TransformerDraft(ck['horizon'], ck['dims'], ck['architecture']).to(device).eval()
                draft.load_state_dict(ck['state_dict'])
                draft.requires_grad_(False)
                self.drafts.append(draft)
        self.trace = {}

    def prompt(self, img, wrist, instruction, state, prev_img=None, prev_wrist=None):
        self.context_state = state.copy()
        images = ([prev_img, prev_wrist] if prev_img is not None else [])+[img, wrist]
        conv = {'conversations': [{'from': 'human', 'value': 'What action should the robot take to '+instruction+'?<|state|>'+'<|image|>'*len(images)}], 'image': images, 'action': [], 'state': state}
        return torch.tensor(self.processor.process_item(conv, training_mode=False), device=self.device, dtype=torch.long)[None]

    @torch.inference_mode()
    def native(self, prompt):
        self.model.init_input_ids = None
        out = self.model.generate(input_ids=prompt, generation_config=self.config,
            training=False, return_dict_in_generate=True, output_hidden_states=False)
        return out.sequences[0, prompt.shape[1]:].tolist(), None

    @torch.inference_mode()
    def prefill(self, prompt):
        self.model.init_input_ids = None
        out = self.model(input_ids=prompt, past_key_values=DynamicCache(), use_cache=True,
                         output_hidden_states=True, return_dict=True)
        return out, context_feature(prompt, out.hidden_states[-1], self.context_state)

    @torch.inference_mode()
    def verify(self, prompt, proposed, prefill):
        from model.chameleon import ChameleonForConditionalGeneration
        n = prompt.shape[1]
        draft = torch.tensor([proposed], device=self.device)
        full = torch.cat([prompt, draft], 1)
        # Preserve official incremental special mask, never substitute generic attention.
        mask = self.model.generate_att_mask_3(full)[:, None, n:, :]
        additive = torch.zeros(mask.shape, device=self.device, dtype=self.model.dtype).masked_fill(~mask, torch.finfo(self.model.dtype).min)
        out = ChameleonForConditionalGeneration.forward(self.model, input_ids=draft,
            attention_mask=additive, past_key_values=prefill.past_key_values,
            position_ids=torch.arange(n, full.shape[1], device=self.device)[None],
            cache_position=torch.arange(n, full.shape[1], device=self.device), use_cache=True, return_dict=True)
        logits = torch.cat([prefill.logits[:, -1:], out.logits[:, :-1]], 1)
        return logits.argmax(-1)[0].tolist(), out.past_key_values

    @torch.inference_mode()
    def recover_original(self, prompt, accepted, cache, boundary_token):
        # Keep verifier KV through the accepted prefix; reuse verifier boundary argmax.
        # Avoids re-forwarding the last accepted token under a different BF16 shape.
        n = prompt.shape[1]
        if isinstance(cache, tuple):
            cache = DynamicCache.from_legacy_cache(cache)
        cache.crop(n + len(accepted))
        full = torch.cat([prompt, torch.tensor([accepted], device=self.device, dtype=torch.long)], 1) if accepted else prompt.clone()
        self.model.init_input_ids = full.clone()
        if int(boundary_token) == 8710:
            return accepted + [8710]
        generated = [int(boundary_token)]
        for _ in range(self.max_new_tokens - len(accepted) - 1):
            out = self.model(input_ids=torch.tensor([[generated[-1]]], device=self.device),
                             past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            t = int(out.logits[0, -1].argmax())
            generated.append(t)
            if t == 8710:
                break
        return accepted + generated

    @torch.inference_mode()
    def spec(self, prompt):
        if len(self.drafts) != 3:
            raise ValueError('Spec requires three independently loaded drafts')
        pre, feature = self.prefill(prompt)
        self.trace = dict(branch_calls=[0, 0, 0], block_calls=0, gates=[],
            reference='candidate-prefix-conditioned frozen target; not independent AR oracle',
            acceptance='absolute explicit-unit per-axis tolerance, atomic 7D',
            bf16_boundary_reuse=True)
        parts = []
        for i, draft in enumerate(self.drafts):
            parts.append(draft(feature[None]).argmax(-1)[0])
            self.trace['branch_calls'][i] += 1
        proposal = [v for row in (torch.cat(parts, -1)+10006).tolist() for v in [10004]+row+[15004]]+[8710]
        pred, cache = self.verify(prompt, proposal, pre)
        self.trace['block_calls'] += 1
        accepted, count = [], 0
        for j in range(self.horizon):
            a, b = proposal[j*9:(j+1)*9], pred[j*9:(j+1)*9]
            ok, why = self.absolute_gate(a, b)
            self.trace['gates'].append(dict(step=j, accepted=ok, **why))
            if not ok:
                break
            accepted += a
            count += 1
        self.trace['accepted_steps'] = count
        if count == self.horizon and pred[-1] == 8710:
            return proposal, count
        boundary = pred[len(accepted)] if len(accepted) < len(pred) else 8710
        self.trace['boundary_token'] = boundary
        return self.recover_original(prompt, accepted, cache, boundary), count

def steps(tokens):
    out, start = [], None
    for i, token in enumerate(tokens):
        if token == 10004:
            start = i
        elif token == 15004 and start is not None:
            row = tokens[start:i+1]
            if len(row) != 9:
                raise ValueError('Malformed official action '+str(row))
            out.append(row)
            start = None
    return out

def commands(tokens, device='cuda'):
    # Preserve official GPU float32 decode (not the gate's scalar decoder).
    import numpy as np
    from model import ChameleonXLLMXForConditionalGeneration_ck_action_head as Target
    from .gate import LOW, HIGH
    action = Target.decode_token_ids_to_actions(None, torch.as_tensor(tokens, device=device)).float().cpu().numpy()
    lo, hi = np.array(LOW), np.array(HIGH)
    return (action+1)/2*(hi-lo+1e-8)+lo
