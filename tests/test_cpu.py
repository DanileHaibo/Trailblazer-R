import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from trailblazer.draft import temporal, load_draft
from trailblazer.engine import Engine, action_positions
from trailblazer.runtime import NoraPolicy, checked_action
from trailblazer.evaluate import identical

ROOT = Path(__file__).resolve().parents[1]


class CPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_temporal_shift_and_padding_both_models(self):
        for N, M in [(5, 2), (10, 5)]:
            previous = np.arange(N*7, dtype=np.float32).reshape(N, 7)
            prior, mask = temporal(np.zeros(8), previous, N, M)
            expected = np.concatenate([previous[M:], np.repeat(previous[-1:], M, axis=0)])
            np.testing.assert_array_equal(prior[8:].reshape(N, 7), expected)
            self.assertEqual(mask.shape, (2+2*N,))
            self.assertEqual(mask[0], 1)
            self.assertEqual(mask[1], M)
            q0, qm = temporal(np.zeros(8), None, N, M)
            self.assertTrue(np.all(q0 == 0) and np.all(qm == 0))

    def test_export_hashes_size_and_no_optimizer(self):
        provenance = json.loads((ROOT/'configs/provenance.json').read_text())
        for name in ['nora', 'pi0']:
            path = ROOT/'weights'/f'{name}_gated.pt'
            self.assertLess(path.stat().st_size, 100*1024**2)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance[name]['export_sha256'])
            model, ck = load_draft(path)
            self.assertEqual(set(ck), {'kind', 'state_dict', 'model_config', 'selected_update'})
            N = ck['model_config']['N']
            prior, mask = temporal(np.zeros(8), np.ones((N, 7)), N, 2 if N == 5 else 5)
            with torch.inference_mode():
                result = model(torch.zeros(1, 4, 2048), torch.from_numpy(prior)[None], torch.zeros(1, 4, dtype=torch.bool), torch.from_numpy(mask)[None])
            self.assertEqual(tuple(result.shape), (1, N, 7))
            self.assertTrue(torch.isfinite(result).all())
            if name == 'nora':
                self.assertEqual(ck['selected_update'], 0)
                np.testing.assert_array_equal(result.numpy()[0], prior[8:].reshape(N, 7))
                self.assertEqual(torch.count_nonzero(model.action_head.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(model.action_head.bias).item(), 0)

    def test_nora_decode_rejects_malformed_not_zeros(self):
        policy = NoraPolicy.__new__(NoraPolicy)
        policy.n = SimpleNamespace(fast_tokenizer=SimpleNamespace(bpe_tokenizer=SimpleNamespace(decode=lambda ids: 'x'*34)))
        with self.assertRaisesRegex(ValueError, 'Malformed teacher chunk'):
            policy.decode([151665, 151645])

    def test_nora_mask_and_invalid_proposal(self):
        p = SimpleNamespace(name='nora', n=SimpleNamespace(fast_tokenizer=SimpleNamespace(bpe_tokenizer=range(2048))))
        self.assertEqual(action_positions(p, [151665, 151666, 151645]), [0, 1])
        for tokens in [[], [151665], [151664, 151645], [153713, 151645]]:
            with self.assertRaises(ValueError):
                action_positions(p, tokens)

    def test_pi0_mask_excludes_wrappers(self):
        tok = SimpleNamespace(encode=lambda text, **kw: [10, 11] if text == 'Action: ' else [12], eos_token_id=1, vocab_size=100)
        p = SimpleNamespace(name='pi0', p=SimpleNamespace(_paligemma_tokenizer=tok, config=SimpleNamespace(fast_skip_tokens=0), action_tokenizer=SimpleNamespace(bpe_tokenizer=range(5))))
        self.assertEqual(action_positions(p, [10, 11, 99, 98, 12, 1]), [2, 3])
        with self.assertRaises(ValueError):
            action_positions(p, [10, 11, 90, 12, 1])

    def test_action_shape_finite(self):
        for value in [np.zeros((4, 7)), np.full((5, 7), np.nan)]:
            with self.assertRaises(ValueError):
                checked_action(value, 5)

    def test_candidate_exception_restores_rope_and_original_cache(self):
        model = SimpleNamespace(rope_deltas=torch.ones(1))
        p = SimpleNamespace(name='nora', N=5, M=2, n=SimpleNamespace(model=model))
        engine = Engine(p, lambda *args: (_ for _ in ()).throw(ValueError('bad draft')), -5.)
        cache = SimpleNamespace(get_seq_length=lambda: 10)
        with patch.object(torch.Tensor, 'cuda', lambda x: x):
            answer = engine.decision(torch.zeros(1, 2, 2048), torch.zeros(1, 2, dtype=torch.bool), cache, torch.zeros(4), None, np.zeros(8), np.zeros((5, 7)))
        self.assertIsNone(answer)
        self.assertFalse(engine.result['accepted'])
        self.assertIn('bad draft', engine.result['candidate_error'])
        self.assertTrue(torch.equal(model.rope_deltas, torch.ones(1)))

    def test_candidate_block_isolated_accept_and_reject_restore_rope(self):
        for threshold, accepted in [(-20., True), (float('inf'), False)]:
            model = SimpleNamespace(rope_deltas=torch.ones(1))
            p = SimpleNamespace(name='nora', N=5, M=2, n=SimpleNamespace(model=model, fast_tokenizer=SimpleNamespace(bpe_tokenizer=range(4))), encode=lambda raw: [151665, 151645], decode=lambda tokens: np.ones((5, 7)))
            draft = lambda *args: torch.ones(1, 5, 7)
            cache = SimpleNamespace(length=10)
            cache.get_seq_length = lambda: cache.length
            def block(tokens, isolated):
                self.assertIsNot(isolated, cache)
                model.rope_deltas = torch.zeros(1)
                return torch.zeros(2, 151666)
            engine = Engine(p, draft, threshold)
            with patch.object(torch.Tensor, 'cuda', lambda x: x):
                answer = engine.decision(torch.zeros(1, 2, 2048), torch.zeros(1, 2, dtype=torch.bool), cache, torch.zeros(151666), block, np.zeros(8), np.zeros((5, 7)))
            self.assertEqual(engine.result['accepted'], accepted)
            self.assertEqual(answer is not None, engine.result['accepted'])
            self.assertTrue(torch.equal(model.rope_deltas, torch.ones(1)))
            self.assertEqual(cache.length, 10)

    def test_instrument_counts_one_prefill_and_rejects_two(self):
        from trailblazer.instrument import Instrument
        class Target(torch.nn.Module):
            def forward(self, **kw):
                return None
        target = Target()
        p = SimpleNamespace(name='nora', n=SimpleNamespace(model=target))
        instrument = Instrument(p)
        def query(twice=False):
            target(input_ids=torch.zeros(1, 3, dtype=torch.long), past_key_values=None)
            if twice:
                target(input_ids=torch.zeros(1, 3, dtype=torch.long), past_key_values=None)
            target(input_ids=torch.zeros(1, 2, dtype=torch.long), past_key_values=SimpleNamespace(get_seq_length=lambda: 3))
            return np.zeros((5, 7)), np.zeros((5, 7)), {}
        with patch('torch.cuda.synchronize', lambda: None):
            row = instrument.run(query)[2]
            self.assertEqual(row['forward_counts'], {'native_prefill': 1, 'candidate_block': 1, 'native_AR': 0, 'draft': 0})
            with self.assertRaisesRegex(RuntimeError, 'Single-prefill'):
                instrument.run(lambda: query(True))

    def test_q0_never_invokes_draft(self):
        answer = (np.zeros((5, 7)), np.zeros((5, 7)), {'tokens': [1]})
        p = SimpleNamespace(query=lambda *args: answer)
        result = Engine(p, None, -5).query({}, 'instruction', None)
        self.assertTrue(identical(answer, result))
        self.assertTrue(result[2]['q0_native'])


if __name__ == '__main__':
    unittest.main()
