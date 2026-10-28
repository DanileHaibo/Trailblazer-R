"""One explicit model/environment per process; outputs never modify historical logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('USE_TF', '0')  # Transformers must not import TensorFlow; image path imports it explicitly.
ROOT = Path(__file__).resolve().parents[1]
SUITES = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']


def digest(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def reset(suite, task, initial):
    import numpy as np
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    t = suite.get_task(task)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file), camera_heights=256, camera_widths=256)
    try:
        env.seed(7); env.reset()
        obs = env.set_init_state(suite.get_task_init_states(task)[initial])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        env.sim.forward()
        env._update_observables(force=True)
        obs = env.env._get_observations()
        obs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        return env, obs, t.language
    except Exception:
        env.close()
        raise


def identical(a, b):
    import numpy as np
    return a[2]['tokens'] == b[2]['tokens'] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['nora', 'pi0'], required=True)
    parser.add_argument('--arm', choices=['native', 'ours'], required=True)
    parser.add_argument('--suite', choices=SUITES, default='libero_object')
    parser.add_argument('--tasks', nargs='+', type=int, default=[0])
    parser.add_argument('--initials', nargs='+', type=int, default=[0])
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--shared-gpu', action='store_true', help='Mark all latency as descriptive, not formal')
    args = parser.parse_args()
    if any(t not in range(10) for t in args.tasks) or any(i not in range(50) for i in args.initials):
        parser.error('Tasks must be 0..9; initials must be 0..49')
    if args.output.exists():
        parser.error('Output exists: use a new directory; no silent overwrite or invalid resume')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for official model execution; CPU tests use unittest')
    torch.set_num_threads(4); torch.manual_seed(7); np.random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Only allow the NumPy constructors required by trusted official LIBERO fixtures.
    # Do not globally disable weights_only or load untrusted pickle files.
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([(np.core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'), np.ndarray, np.dtype, type(np.dtype('float64'))])
    from libero.libero import benchmark
    from .runtime import NoraPolicy, PiPolicy, state
    from .draft import load_draft
    from .engine import Engine
    config = json.loads((ROOT/'configs/protocol.json').read_text())
    assets = json.loads(args.assets.read_text())
    p = NoraPolicy(assets, args.suite, config) if args.model == 'nora' else PiPolicy(assets)
    draft = None
    if args.arm == 'ours':
        draft, _ = load_draft(ROOT/'weights'/f'{args.model}_gated.pt', 'cuda')
    engine = Engine(p, draft, config['thresholds'][args.model]) if draft is not None else None
    from .instrument import Instrument
    instrument = Instrument(p, draft)
    raw_query = p.query
    p.query = lambda *a, **kw: instrument.run(lambda: raw_query(*a, **kw))
    if engine is not None:
        raw_engine_query = engine.query
        engine.query = lambda *a, **kw: instrument.run(lambda: raw_engine_query(*a, **kw))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    args.output.mkdir(parents=True)
    def dump(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    dump('protocol.json', dict(model=args.model, arm=args.arm, suite=args.suite, tasks=args.tasks, initials=args.initials, N=p.N, M=p.M, seed=7, settling=10, max_actual_control_steps=500, synchronized_reset=True, shared_gpu=args.shared_gpu, latency_valid_for_formal_comparison=False, latency_note='Descriptive only; no matched-frame timing study in this release runner', approximate=args.arm=='ours', assets=assets))
    # A new reset prevents any warmup/preflight action leakage into scored rollouts.
    env, obs, language = reset(suite, args.tasks[0], args.initials[0])
    try:
        a = p.query(obs, language); b = p.query(obs, language)
        checks = {'native_repeat_exact': identical(a, b)}
        if draft is not None:
            rejected = instrument.run(lambda: Engine(p, draft, float('inf')).query(obs, language, a[1]))
            q0 = engine.query(obs, language, None)
            checks.update(forced_reject_exact=identical(a, rejected), q0_exact=identical(a, q0))
        dump('preflight.json', checks)
        if not all(checks.values()):
            raise RuntimeError('Preflight exact fallback failed; no scored episodes run')
    finally:
        env.close()
    if args.preflight_only:
        return
    episodes = []
    for task in args.tasks:
        for initial in args.initials:
            env, obs, language = reset(suite, task, initial)
            row = dict(task=task, initial=initial, initial_digest=digest(suite.get_task_init_states(task)[initial]), reset_hashes={k: digest(obs[k]) for k in ['agentview_image', 'robot0_eye_in_hand_image']}, state_hash=digest(state(obs)), success=False, control_steps=0, queries=0, accepted=0, error=None)
            previous = None
            try:
                with (args.output/f'task{task}_init{initial}.jsonl').open('x') as stream:
                    while row['control_steps'] < 500:
                        cmd, y, detail = p.query(obs, language) if engine is None else engine.query(obs, language, previous)
                        stream.write(json.dumps(detail, allow_nan=False)+'\n'); stream.flush()
                        row['queries'] += 1; row['accepted'] += int(detail.get('accepted', False))
                        for action in cmd[:min(p.M, 500-row['control_steps'])]:
                            obs, _, done, _ = env.step(action.tolist()); row['control_steps'] += 1
                            if done or env.check_success():
                                row['success'] = True
                                break
                        previous = y.copy()
                        if row['success']:
                            break
            except Exception as exc:
                row['error'] = repr(exc)
                row['success'] = False
                # Invalid target decoding is a failed rollout, not a synthetic zero action.
                if not (isinstance(exc, ValueError) and str(exc).startswith('Malformed teacher chunk:')) and not (isinstance(exc, AssertionError) and 'Token sequence does not start with' in str(exc)):
                    dump(f'task{task}_init{initial}_episode.json', row)
                    raise
            finally:
                env.close()
            episodes.append(row)
            dump(f'task{task}_init{initial}_episode.json', row)
            dump('summary.json', dict(episodes=len(episodes), successes=sum(e['success'] for e in episodes), errors=sum(e['error'] is not None for e in episodes), complete=len(episodes)==len(args.tasks)*len(args.initials)))


if __name__ == '__main__':
    main()
