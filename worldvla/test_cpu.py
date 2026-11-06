"""CPU-only codec, gate, control-flow and real checkpoint tests.

Run: python -m unittest worldvla.test_cpu -v
"""
import unittest
from pathlib import Path
from .contract import valid, decode_step
from .engine import AbsoluteGate, DEFAULT_GATE, query


def step(token=10133):
    return [10004]+[token]*7+[15004]

class CPUContracts(unittest.TestCase):
    def test_gate_units_boundaries_and_grip(self):
        gate = AbsoluteGate(DEFAULT_GATE)
        a = step()
        self.assertTrue(gate(a, a)[0])
        for dim in range(6):
            for offset in range(1, 50):
                b = a.copy(); b[dim+1] += offset
                c, r = decode_step(a), decode_step(b)
                error = abs(c[dim]-r[dim])*(.05 if dim < 3 else .5)
                passed, reason = gate(a, b)
                self.assertAlmostEqual(reason['absolute_error_controller_goal'][dim], error)
                self.assertEqual(passed, error <= (.01 if dim < 3 else .05))
        b = a.copy(); b[7] = 10006
        self.assertFalse(gate(a, b)[0])
        self.assertFalse(gate([0], a)[0])
        with self.assertRaises(ValueError):
            AbsoluteGate(dict(DEFAULT_GATE, xyz_threshold=float('nan')))

    def test_single_block_first_rejection_and_eos_recovery(self):
        proposal = step()*5+[8710]
        for reject in range(7):
            refs = proposal.copy()
            if reject < 5:
                refs[reject*9+1] = 10261
            elif reject == 5:
                refs[-1] = 0
            class Draft:
                calls = 0
                def propose(self, feature):
                    self.calls += 1; return proposal
            class Runtime:
                horizon = 5
                prefills = 0; blocks = 0; recoveries = 0
                def prefill(self, prompt):
                    self.prefills += 1; return 'prefill', 'feature'
                def block(self, prompt, proposed, prefill):
                    self.blocks += 1; return refs, 'cache'
                def recover(self, prompt, accepted, cache):
                    self.recoveries += 1
                    self.accepted = accepted
                    return accepted+step()*(5-len(accepted)//9)+[8710]
            rt, draft = Runtime(), Draft()
            tokens, stats = query(rt, None, draft, AbsoluteGate(DEFAULT_GATE))
            self.assertEqual((rt.prefills, rt.blocks, draft.calls), (1, 1, 1))
            self.assertEqual(stats['accepted_steps'], min(reject, 5))
            self.assertEqual(rt.recoveries, int(reject < 6))
            if reject < 6:
                self.assertEqual(rt.accepted, proposal[:min(reject, 5)*9])
            valid(tokens, 5)

    def test_real_checkpoints_three_branches_once(self):
        import torch
        from .rich import RichDraft
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]/'weights/worldvla'
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
            path = root/f'{suite}.pt'
            self.assertLess(path.stat().st_size, 100_000_000)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model = RichDraft(ck['horizon']).eval(); model.load_state_dict(ck['state'])
            counts = [0, 0, 0]
            handles = [head.register_forward_hook(lambda m, a, o, i=i: counts.__setitem__(i, counts[i]+1)) for i, head in enumerate(model.heads)]
            tokens = model.propose((torch.zeros(10, 4096), torch.zeros(8)))
            self.assertEqual(counts, [1, 1, 1])
            self.assertEqual(len(valid(tokens, ck['horizon'])), ck['horizon'])
            for h in handles:
                h.remove()

if __name__ == '__main__':
    unittest.main()
