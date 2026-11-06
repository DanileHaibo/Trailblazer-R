"""Original worldvla_transformer_absolute: one block, first-reject AR recovery.

Threshold units are controller goal increments, not measured displacement or
rotation geodesic distance. Teacher-forced references are candidate-prefix
conditional and are NOT an independent native AR oracle.
"""
import math
from .contract import decode_step

DEFAULT_GATE = dict(xyz_unit='m_goal_delta', rotation_unit='rad_axisangle_delta_component',
                    gripper_unit='binary_sign', xyz_threshold=.01, rotation_threshold=.05,
                    controller_scale_xyz=.05, controller_scale_rotation=.5, control_delta=True)

class AbsoluteGate:
    def __init__(self, config):
        self.config = dict(config)
        if config.get('xyz_unit') != 'm_goal_delta' or config.get('rotation_unit') != 'rad_axisangle_delta_component' or config.get('gripper_unit') != 'binary_sign':
            raise ValueError('Explicit controller-goal units required: m_goal_delta, rad_axisangle_delta_component, binary_sign')
        self.xyz = config['xyz_threshold']; self.rot = config['rotation_threshold']
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in [self.xyz, self.rot]):
            raise ValueError('finite nonnegative absolute thresholds required')
        if config.get('controller_scale_xyz') != .05 or config.get('controller_scale_rotation') != .5 or config.get('control_delta') is not True:
            raise ValueError('Must confirm OSC_POSE input +/-1, output xyz +/-0.05m and rotation +/-0.5rad, control_delta=true')

    def __call__(self, candidate, reference):
        try:
            c, r = decode_step(candidate), decode_step(reference)
        except ValueError:
            return False, {'invalid': True}
        errors = [abs(max(-1, min(1, a))-max(-1, min(1, b)))*(.05 if i < 3 else .5) for i, (a, b) in enumerate(zip(c[:6], r[:6]))]
        dims = [e <= (self.xyz if i < 3 else self.rot) for i, e in enumerate(errors)]
        grip = (c[6] > 0) == (r[6] > 0)
        passed = all(dims) and grip and candidate[0] == reference[0] and candidate[-1] == reference[-1]
        return passed, {'absolute_error_controller_goal': errors, 'xyz_unit': 'm_goal_delta', 'rotation_unit': 'rad_axisangle_delta_component', 'gripper_unit': 'binary_sign', 'gripper_sign': grip, 'dimension_pass': dims+[grip]}

def query(rt, prompt, draft, gate):
    prefill, feature = rt.prefill(prompt)
    proposal = list(draft.propose(feature))
    if len(proposal) != rt.horizon*9+1 or proposal[-1] != 8710:
        raise ValueError('fresh full-horizon chunk required')
    for i in range(rt.horizon):
        decode_step(proposal[i*9:i*9+9])
    refs, cache = rt.block(prompt, proposal, prefill)
    if len(refs) != len(proposal):
        raise RuntimeError('shifted-logit mismatch')
    accepted = []; groups = []
    for step in range(rt.horizon):
        start = step*9
        passed, reason = gate(proposal[start:start+9], refs[start:start+9])
        groups.append(dict(step=step, accepted=passed, **reason))
        if not passed:
            break
        accepted.extend(proposal[start:start+9])
    recovery = len(accepted) != rt.horizon*9 or refs[-1] != 8710
    tokens = rt.recover(prompt, accepted, cache) if recovery else accepted+[8710]
    return tokens, dict(accepted_steps=len(accepted)//9, proposed_steps=rt.horizon, groups=groups,
                        verifier_calls=1, draft_calls=1, ar_recovery=recovery,
                        reference_semantics='candidate-prefix-conditioned teacher forced; NOT independent native AR',
                        proposal=proposal, reference=refs)
