"""Single native prefill, isolated approximate candidate scoring, original-cache fallback."""
import copy
import math
import time
import numpy as np
import torch
from .draft import temporal


def sync():
    torch.cuda.synchronize()


def action_positions(p, tokens):
    if p.name == 'nora':
        if not tokens or tokens[-1] != 151645:
            raise ValueError('NORA proposal must end in EOS')
        ids = [t-151665 for t in tokens[:-1]]
        if not ids or not all(0 <= t < len(p.n.fast_tokenizer.bpe_tokenizer) for t in ids):
            raise ValueError('Invalid FAST proposal IDs')
        return list(range(len(ids)))
    tok = p.p._paligemma_tokenizer
    prefix = tok.encode('Action: ', add_special_tokens=False)
    suffix = tok.encode('|', add_special_tokens=False) + [tok.eos_token_id]
    if tokens[:len(prefix)] != prefix or tokens[-len(suffix):] != suffix:
        raise ValueError('Invalid PI0 action wrapper')
    positions = list(range(len(prefix), len(tokens)-len(suffix)))
    cap = tok.vocab_size-1-p.p.config.fast_skip_tokens
    if not positions or not all(0 <= cap-tokens[i] < len(p.p.action_tokenizer.bpe_tokenizer) for i in positions):
        raise ValueError('Invalid PI0 FAST IDs')
    return positions


class Engine:
    def __init__(self, policy, draft, threshold):
        self.p, self.draft, self.threshold = policy, draft, threshold
        self.result = {}

    def decision(self, memory, padding, cache, firstlogits, blockfn, state, previous):
        p = self.p
        model = p.n.model if p.name == 'nora' else None
        rope = getattr(model, 'rope_deltas', None)
        rope = None if rope is None else rope.detach().clone()
        length = cache.get_seq_length()
        self.result = {'accepted': False, 'candidate_error': None, 'ranking_score': None}
        try:
            prior, mask = temporal(state, previous, p.N, p.M)
            raw = self.draft(memory, torch.from_numpy(prior)[None].cuda(), padding, torch.from_numpy(mask)[None].cuda())[0].float().cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite draft')
            tokens = p.encode(raw)
            positions = action_positions(p, tokens)
            candidate = p.decode(tokens)
            isolated = copy.deepcopy(cache)
            logits = blockfn(tokens, isolated)
            logits = torch.cat([firstlogits.reshape(1, -1), logits[:-1]], 0)
            target = torch.tensor([tokens[i] for i in positions], device=logits.device)
            score = float(torch.log_softmax(logits[positions].float(), -1).gather(1, target[:, None]).mean())
            self.result.update(ranking_score=score, accepted=score >= self.threshold)
            if self.result['accepted']:
                self.candidate = candidate
                return tokens
        except Exception as exc:
            self.result.update(accepted=False, candidate_error=repr(exc))
        finally:
            if model is not None:
                model.rope_deltas = rope
            if cache.get_seq_length() != length:
                raise RuntimeError('Candidate corrupted original cache')
        return None

    @torch.inference_mode()
    def query(self, obs, instruction, previous=None):
        p = self.p
        if previous is None:
            cmd, y, row = p.query(obs, instruction)
            row.update(accepted=False, q0_native=True)
            return cmd, y, row
        sync(); start = time.perf_counter()
        from .runtime import state
        if p.name == 'nora':
            from transformers import RepetitionPenaltyLogitsProcessor
            model = p.n.model
            inputs = p.inputs(obs, instruction)
            prompt = inputs['input_ids']; plen = prompt.shape[1]
            model.rope_deltas = None
            captured = []
            handle = model.model.register_forward_hook(lambda mod, args, out: captured.append(out.last_hidden_state.detach()))
            try:
                out = model(**inputs, use_cache=True, return_dict=True, cache_position=torch.arange(plen, device='cuda'))
            finally:
                handle.remove()
            cache = out.past_key_values; firstlogits = out.logits[0, -1].clone()
            processor = RepetitionPenaltyLogitsProcessor(model.generation_config.repetition_penalty)
            def choose(logits, generated):
                prefix = torch.cat([prompt, torch.tensor([generated], device='cuda', dtype=torch.long)], 1)
                return int(processor(prefix, logits.float()[None].clone()).argmax(-1).item())
            first = choose(firstlogits, [])
            def blockfn(tokens, isolated):
                cached = isolated.get_seq_length()
                out = model(input_ids=torch.tensor([tokens], device='cuda'), past_key_values=isolated, use_cache=True, return_dict=True, attention_mask=torch.ones(1, cached+len(tokens), device='cuda', dtype=torch.long), cache_position=torch.arange(cached, cached+len(tokens), device='cuda'))
                return out.logits[0]
            accepted = self.decision(captured[0], ~inputs['attention_mask'].bool(), cache, firstlogits, blockfn, state(obs), previous)
            if accepted is not None:
                tokens, y = accepted, self.candidate
            else:
                tokens = [first]
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                while len(tokens) < p.limit and tokens[-1] not in eos:
                    cached = cache.get_seq_length()
                    out = model(input_ids=torch.tensor([[tokens[-1]]], device='cuda'), past_key_values=cache, use_cache=True, return_dict=True, attention_mask=torch.ones(1, cached+1, device='cuda', dtype=torch.long), cache_position=torch.arange(cached, cached+1, device='cuda'))
                    cache = out.past_key_values
                    tokens.append(choose(out.logits[0, -1], tokens))
                y = p.decode(tokens)
            cmd = p.commands(y)
        else:
            original = p.p.model.sample_actions_fast_kv_cache
            def sampler(images, img_masks, tokens, masks, max_decoding_steps=None, temperature=0.):
                m = p.p.model; v = m.paligemma_with_expert
                limit = max_decoding_steps or m.config.max_action_tokens
                bos = torch.full((1, 1), m._paligemma_tokenizer.bos_token_id, dtype=torch.long, device='cuda')
                emb, pad, att, _, _ = m.embed_prefix_fast(images, img_masks, torch.cat([tokens, bos], 1), torch.cat([masks, torch.ones(1, 1, dtype=torch.bool, device='cuda')], 1))
                emb = emb.to(v.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
                (hidden, _), cache = v.forward(attention_mask=m._prepare_attention_masks_4d(att, dtype=emb.dtype), position_ids=pad.long().cumsum(1)-1, past_key_values=None, inputs_embeds=[emb, None], use_cache=True, adarms_cond=[None, None])
                firstlogits = v.paligemma.lm_head(hidden[:, -1:])[0, 0]
                def blockfn(candidates, isolated):
                    n = len(candidates)
                    be = v.embed_language_tokens(torch.tensor([candidates], device='cuda'))
                    be = (be*math.sqrt(be.shape[-1])).to(emb.dtype)
                    hist = torch.cat([pad, torch.ones(1, n, dtype=torch.bool, device='cuda')], 1)
                    allowed = hist[:, None, :].expand(-1, n, -1).clone()
                    allowed[:, :, pad.shape[1]:] = torch.tril(torch.ones(n, n, dtype=torch.bool, device='cuda'))[None]
                    pos = pad.long().sum(1, keepdim=True)+torch.arange(n, device='cuda')[None]
                    (h, _), _ = v.forward(attention_mask=m._prepare_attention_masks_4d(allowed, dtype=emb.dtype), position_ids=pos, past_key_values=isolated, inputs_embeds=[be, None], use_cache=True, adarms_cond=[None, None])
                    return v.paligemma.lm_head(h)[0]
                accepted = self.decision(hidden, ~pad.bool(), cache, firstlogits, blockfn, state(obs), previous)
                if accepted is not None:
                    return torch.tensor([accepted], device='cuda')
                generated = [int(firstlogits.argmax().item())]; current = pad
                for _ in range(1, limit):
                    be = v.embed_language_tokens(torch.tensor([[generated[-1]]], device='cuda'))
                    be = (be*math.sqrt(be.shape[-1])).to(emb.dtype)
                    current = torch.cat([current, torch.ones(1, 1, dtype=torch.bool, device='cuda')], 1)
                    (h, _), cache = v.forward(attention_mask=m._prepare_attention_masks_4d(current.unsqueeze(1), dtype=be.dtype), position_ids=(current.sum(1, keepdim=True)-1).long(), past_key_values=cache, inputs_embeds=[be, None], use_cache=True, adarms_cond=[None, None])
                    generated.append(int(v.paligemma.lm_head(h[:, -1:]).argmax(-1).item()))
                return torch.tensor([generated], device='cuda')
            p.p.model.sample_actions_fast_kv_cache = sampler
            try:
                cmd, y, row = p.query(obs, instruction)
                tokens = row['tokens']
            finally:
                p.p.model.sample_actions_fast_kv_cache = original
        sync()
        return cmd, np.asarray(y, np.float32), dict(tokens=tokens, **self.result, policy_ms=(time.perf_counter()-start)*1000)
