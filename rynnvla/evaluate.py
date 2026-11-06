"""Portable fair_v5/200-pair final_v7 evaluation, parameterized without training.
Default plan: 4 suites x 10 tasks x initials 10..14 = 200 paired trials.
Not the standard 500 trials per suite. No automatic retries or oracle calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
RELEASE = Path(__file__).resolve().parents[1]

def indices(value):
    result = []
    for part in value.split(','):
        if ':' in part:
            bounds = [int(x) for x in part.split(':')]
            if len(bounds) not in (2, 3):
                raise argparse.ArgumentTypeError('Use comma IDs or start:stop[:step]')
            result.extend(range(*bounds))
        else:
            result.append(int(part))
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('Expected unique nonnegative IDs')
    return result

def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-path', '--base', dest='base_path', required=True, help='Suite checkpoint or parent containing four suite directories; never bundled')
    ap.add_argument('--official-repo', required=True, help='Pinned external RynnVLA-002 checkout')
    ap.add_argument('--libero-repo', help='External LIBERO checkout; omit if installed')
    ap.add_argument('--tokenizer-path', required=True, help='Official shared base_model text tokenizer directory')
    ap.add_argument('--image-tokenizer-path', required=True, help='Official shared chameleon/tokenizer directory')
    ap.add_argument('--suite', nargs='+', choices=SUITES+['all'], default=['all'])
    ap.add_argument('--checkpoint', type=Path, default=RELEASE/'weights/rynnvla', help='Directory with suite_{xyz,rot,grip}_best.pt')
    ap.add_argument('--config', type=Path, default=RELEASE/'configs/rynnvla.json')
    ap.add_argument('--output', type=Path, required=True, help='New output directory; refuses overwrite')
    ap.add_argument('--tasks', type=indices, default=indices('0:10'))
    ap.add_argument('--initials', type=indices, default=indices('10:15'))
    ap.add_argument('--mode', nargs='+', choices=['native', 'spec'], default=['native', 'spec'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=20261104)
    ap.add_argument('--plan-only', action='store_true', help='Validate arguments and print plan without torch/backend/GPU imports')
    return ap

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

def append(path, obj):
    with path.open('a') as stream:
        stream.write(json.dumps(obj)+'\n')

def make_plan(args):
    suites = SUITES if 'all' in args.suite else args.suite
    if len(set(suites)) != len(suites) or len(set(args.mode)) != len(args.mode):
        raise ValueError('Duplicate suites/modes are not allowed')
    plan = []
    for suite in suites:
        for pairindex, (task, initial) in enumerate((t, i) for t in args.tasks for i in args.initials):
            order = ['native', 'spec'] if pairindex % 2 == 0 else ['spec', 'native']
            plan.append(dict(suite=suite, task=task, init=initial,
                             order=[m for m in order if m in args.mode], pairindex=pairindex))
    return plan

def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if args.plan_only:
        print(json.dumps(dict(pairs=len(plan), episodes=sum(len(p['order']) for p in plan), plan=plan), indent=2))
        return
    # Re-exec before importing torch so Python's hash seed really takes effect.
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        import sys
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.execv(sys.executable, [sys.executable, '-m', 'rynnvla.evaluate'] + (list(argv) if argv is not None else sys.argv[1:]))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # No CUDA_VISIBLE_DEVICES override: caller controls GPU selection.
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    save(args.output/'test_plan.json', plan)
    from .backend import configure, make_processor
    configure(args.official_repo, args.libero_repo)
    import numpy as np
    import torch
    from PIL import Image
    from libero.libero import benchmark
    from libero_util.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action, quat2axisangle
    from .runtime import Runtime, commands, steps
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    def seed():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    def sync():
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
    def digest(value):
        return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    def extract(obs):
        return (Image.fromarray(get_libero_image(obs, 256)),
                Image.fromarray(get_libero_image(obs, 256, 'robot0_eye_in_hand_image')),
                np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']]))
    seed()
    save(args.output/'freeze.json', dict(seed=args.seed, env_seed=0, pythonhash=os.environ['PYTHONHASHSEED'],
        precision=torch.get_float32_matmul_precision(), matmul_tf32=False, cudnn_tf32=False,
        cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True,
        deterministic_warn_only=True, cublas=os.environ['CUBLAS_WORKSPACE_CONFIG'], torch=torch.__version__,
        history_update='every environment step, not every chunk', warmup_excluded=True,
        pairs=len(plan), episodes=sum(len(p['order']) for p in plan), not_standard500_per_suite=True,
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}))
    save(args.output/'gate_config.json', config['gate'])
    processor = make_processor(args.tokenizer_path, args.image_tokenizer_path, args.device)
    allrows = []
    for suite in dict.fromkeys(p['suite'] for p in plan):
        seed()
        rt = Runtime(suite, base_path=args.base_path, processor=processor, checkpoint_dir=args.checkpoint,
                     gate_config=config['gate'], mode='spec' if 'spec' in args.mode else 'native', device=args.device)
        benchmark_suite = benchmark.get_benchmark_dict()[suite]()
        component, phase, counts = {}, ['other'], {}
        for name in ['prefill', 'verify', 'recover_original', 'native']:
            original = getattr(rt, name)
            def wrapped(*a, _orig=original, _name=name, **kw):
                old = phase[0]; phase[0] = _name; sync(); start = time.perf_counter()
                try:
                    return _orig(*a, **kw)
                finally:
                    sync(); component[_name+'_ms'] = component.get(_name+'_ms', 0)+(time.perf_counter()-start)*1000
                    phase[0] = old
            setattr(rt, name, wrapped)
        def count_forward(model, inputs):
            counts[phase[0]] = counts.get(phase[0], 0)+1
        handles = [rt.model.register_forward_pre_hook(count_forward)]
        starts = {}
        for i, draft in enumerate(rt.drafts):
            def pre(model, inputs, i=i):
                sync(); starts[i] = time.perf_counter()
            def post(model, inputs, outputs, i=i):
                sync(); component['draft'+str(i)+'_ms'] = (time.perf_counter()-starts[i])*1000
            handles.extend([draft.register_forward_pre_hook(pre), draft.register_forward_hook(post)])
        for pair in [p for p in plan if p['suite'] == suite]:
            tid, initid = pair['task'], pair['init']
            for mode in pair['order']:
                seed(); env = None; error = None; success = False; executed = 0; query = 0; totalms = 0.; warmup_error = None
                try:
                    env, instruction = get_libero_env(benchmark_suite.get_task(tid), 256)
                    initial = benchmark_suite.get_task_init_states(tid)[initid]
                    def reset(label):
                        seed(); env.seed(0); env.reset(); obs = env.set_init_state(initial)
                        for _ in range(10):
                            obs, _, _, _ = env.step(get_libero_dummy_action())
                        im, wr, state = extract(obs)
                        append(args.output/'resets.jsonl', dict(suite=suite, task=tid, init=initid, mode=mode, label=label,
                            state_hash=digest(env.env.sim.get_state().flatten()), image_hash=digest(im), wrist_hash=digest(wr), proprio=state.tolist(), env_seed=0))
                        return obs
                    obs = reset('warmup'); im, wr, state = extract(obs)
                    prompt = rt.prompt(im, wr, instruction, state)
                    rt.native(prompt) if mode == 'native' else rt.spec(prompt)
                    sync()
                    obs = reset('rollout'); prev = prevw = None; queue = []; active = None
                    for t in range({'libero_spatial':220, 'libero_object':280, 'libero_goal':300, 'libero_10':520}[suite]):
                        sync(); start = time.perf_counter(); im, wr, state = extract(obs); sync()
                        extractms = (time.perf_counter()-start)*1000
                        if not queue:
                            component.clear(); counts.clear(); rt.trace = {}
                            q = dict(suite=suite, task=tid, init=initid, mode=mode, query=query, step=t,
                                image_hash=digest(im), wrist_hash=digest(wr), simstate_hash=digest(env.env.sim.get_state().flatten()), proprio=state.tolist())
                            sync(); startq = time.perf_counter(); infer = None
                            try:
                                start = time.perf_counter(); prompt = rt.prompt(im, wr, instruction, state, prev, prevw); sync()
                                component['processor_ms'] = (time.perf_counter()-start)*1000
                                q.update(prompt_tokens=prompt[0].tolist(), prompt_hash=digest(prompt.cpu().numpy()))
                                infer = time.perf_counter()
                                tokens, _ = rt.native(prompt) if mode == 'native' else rt.spec(prompt)
                                q.update(output_tokens=tokens, output_length=len(tokens), eos_count=tokens.count(8710), last_token=tokens[-1] if tokens else None)
                                chunk = steps(tokens)
                                assert len(chunk) == rt.horizon and tokens[-1] == 8710, 'invalid action chunk'
                                queue = [commands(a[1:8], args.device) for a in chunk]
                                q.update(chunk_steps=len(queue), status='passed')
                                if mode == 'spec':
                                    assert rt.trace['branch_calls'] == [1, 1, 1] and rt.trace['block_calls'] == 1
                            except Exception:
                                q.update(status='failed', error=traceback.format_exc()); raise
                            finally:
                                sync(); elapsed = (time.perf_counter()-startq)*1000+extractms; totalms += elapsed
                                q.update(policy_total_ms=elapsed, components_exclusive_ms=dict(component, extract_ms=extractms),
                                    policy_residual_ms=elapsed-sum(component.values())-extractms,
                                    inference_inclusive_ms=(time.perf_counter()-infer)*1000 if infer is not None else None,
                                    model_forward_counts=dict(counts), ar_forward_count=counts.get('native', 0)+counts.get('recover_original', 0),
                                    trace=json.loads(json.dumps(rt.trace)) if mode == 'spec' else None, trace_partial=q.get('status') != 'passed')
                                append(args.output/'queries.jsonl', q); query += 1; active = query-1
                        action = queue.pop(0)
                        obs, _, done, _ = env.step(action.tolist()); executed += 1
                        # Critical history fix from fair_v5: update at EVERY environment step.
                        prev, prevw = im, wr
                        append(args.output/'executions.jsonl', dict(suite=suite, task=tid, init=initid, mode=mode, query=active, step=t, action=action.tolist(), done=bool(done)))
                        if done:
                            success = True; break
                except Exception:
                    error = traceback.format_exc()
                    # Hardware faults stop immediately; never retry or commit a partial pair.
                    if any(x in error.lower() for x in ['cuda error', 'device-side assert', 'out of memory', 'illegal memory', 'cublas_status', 'cudnn_status']):
                        raise
                finally:
                    if env is not None:
                        env.close()
                    row = dict(suite=suite, task=tid, init=initid, mode=mode, success=success, error=error,
                        warmup_error=warmup_error, queries=query, executed_steps=executed, policy_total_ms=totalms)
                    allrows.append(row); append(args.output/'episodes.jsonl', row)
            append(args.output/'committed_pairs.jsonl', dict(pair, time=time.time()))
        for handle in handles:
            handle.remove()
        del rt
        import gc
        gc.collect(); torch.cuda.empty_cache()
    save(args.output/'summary.json', summarize(allrows))
    save(args.output/'completed.json', dict(complete=True, pairs=len(plan)))

def summarize(rows):
    result = {}
    for suite in dict.fromkeys(r['suite'] for r in rows):
        result[suite] = {}
        for mode in dict.fromkeys(r['mode'] for r in rows):
            group = [r for r in rows if r['suite'] == suite and r['mode'] == mode]
            result[suite][mode] = dict(episodes=len(group), successes=sum(r['success'] for r in group),
                exceptions=sum(bool(r['error']) for r in group), policy_total_ms=sum(r['policy_total_ms'] for r in group),
                executed_steps=sum(r['executed_steps'] for r in group), queries=sum(r['queries'] for r in group))
    return result

if __name__ == '__main__':
    main()
