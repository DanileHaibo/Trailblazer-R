"""Explicit absolute-unit atomic seven-dimensional acceptance gate."""
import json
import math
from pathlib import Path

LOW = [-.9375, -.9375, -.9375, -.24214286, -.375, -.36428571, -1.]
HIGH = [.9375, .9375, .9375, .34821429, .375, .375, 1.]

def decode_command(tokens):
    if len(tokens) != 7 or any(not 10006 <= int(t) <= 10261 for t in tokens):
        raise ValueError('Expected seven official native action tokens')
    centers = [-1 + (2 * min(int(t)-10006, 254)+1)/255 for t in tokens]
    return [(v+1)/2*(hi-lo+1e-8)+lo for v, lo, hi in zip(centers, LOW, HIGH)]

class AbsoluteGate:
    def __init__(self, config):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        self.config = config
        self.thresholds, self.scales, self.units = [], [], []
        for group, n, physical in [('xyz', 3, 'm'), ('rot', 3, 'rad'), ('grip', 1, None)]:
            entry = config[group]
            unit = entry['unit']
            if group == 'grip' and entry.get('mode') == 'switch_sign_only':
                if unit != 'command_units':
                    raise ValueError('Switch-only grip must use command_units')
                self.thresholds += [None]; self.scales += [1.]; self.units += [unit]
                continue
            tol = entry['absolute_tolerance']
            tol = [tol]*n if isinstance(tol, (int, float)) else list(tol)
            if len(tol) != n or any(not math.isfinite(t) or t < 0 for t in tol):
                raise ValueError('Finite nonnegative absolute tolerance required: '+group)
            if unit == 'command_units':
                scale = [1.]*n
            elif physical and unit == physical:
                mapping = entry['verified_controller_mapping']
                if mapping.get('confirmed') is not True or not mapping.get('source_evidence'):
                    raise ValueError('Physical units require confirmed mapping and evidence')
                scale = list(mapping['physical_per_command'])
                if len(scale) != n or any(not math.isfinite(s) or s <= 0 for s in scale):
                    raise ValueError('Invalid physical/controller scale')
                if group == 'rot' and mapping.get('metric') != 'axis_angle_component_delta':
                    raise ValueError('Specify axis_angle_component_delta, not geodesic error')
            else:
                raise ValueError('Unverified units: '+group)
            self.thresholds += tol; self.scales += scale; self.units += [unit]*n
        if config['grip'].get('require_same_open_close') is not True:
            raise ValueError('Gripper agreement must be explicitly enabled')

    def __call__(self, a, b):
        if len(a) != 9 or len(b) != 9 or a[0] != 10004 or b[0] != 10004 or a[-1] != 15004 or b[-1] != 15004:
            return False, {'protocol': False}
        try:
            x, y = decode_command(a[1:8]), decode_command(b[1:8])
        except ValueError:
            return False, {'protocol': False}
        error = [abs(v-w)*s for v, w, s in zip(x, y, self.scales)]
        axes = [True if t is None else e <= t+1e-8 for e, t in zip(error, self.thresholds)]
        sign = lambda v: (v > 0)-(v < 0)
        grip_match = sign(x[6]) == sign(y[6])
        axes[6] = axes[6] and grip_match
        reasons = [name for name, ok in zip(['x_m','y_m','z_m','rx_rad','ry_rad','rz_rad','gripper_sign'], axes) if not ok]
        return all(axes), dict(protocol=True, rejection_reasons=reasons, absolute_error=error,
            units=self.units, absolute_tolerance=self.thresholds, axes=axes,
            gripper_open_close_match=grip_match, atomic_7d=True)
