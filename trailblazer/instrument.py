"""Count actual target forwards uniformly for native and approximate paths."""
import time
import torch


class Instrument:
    def __init__(self, policy, draft=None):
        self.policy = policy
        self.depth = 0
        self.counts = None
        target = policy.n.model if policy.name == 'nora' else policy.p.model.paligemma_with_expert
        original = target.forward
        def forward(*args, **kwargs):
            cache = kwargs.get('past_key_values')
            embeds = kwargs.get('inputs_embeds')
            if isinstance(embeds, (list, tuple)):
                embeds = embeds[0]
            ids = kwargs.get('input_ids')
            span = ids.shape[1] if ids is not None else (embeds.shape[1] if embeds is not None else 1)
            empty = cache is None or cache.get_seq_length() == 0
            kind = 'native_prefill' if empty else ('candidate_block' if span > 1 else 'native_AR')
            if self.depth:
                self.counts[kind] += 1
            return original(*args, **kwargs)
        target.forward = forward
        if draft is not None:
            draft.register_forward_hook(self._draft_hook)

    def _draft_hook(self, *args):
        if self.depth:
            self.counts['draft'] += 1

    def run(self, fn):
        if self.depth:
            return fn()
        self.counts = dict(native_prefill=0, candidate_block=0, native_AR=0, draft=0)
        torch.cuda.synchronize(); start = time.perf_counter()
        self.depth += 1
        try:
            cmd, y, row = fn()
            torch.cuda.synchronize()
            row['policy_ms'] = (time.perf_counter()-start)*1000
            row['forward_counts'] = dict(self.counts)
            if self.counts['native_prefill'] != 1:
                raise RuntimeError('Single-prefill contract violation: '+repr(self.counts))
            return cmd, y, row
        finally:
            self.depth -= 1
