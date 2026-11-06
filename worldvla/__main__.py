"""Canonical closed-loop CLI. Heavy imports occur only after --help parsing."""
import argparse
import hashlib
import json
from pathlib import Path

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')

def indices(text):
    result = []
    for item in text.split(','):
        if '-' in item:
            a, b = map(int, item.split('-', 1)); result.extend(range(a, b+1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError('expected unique nonnegative indices, e.g. 0-9 or 0,2,4')
    return result

def parser():
    p = argparse.ArgumentParser(description='WorldVLA canonical rich 3-draft / single block / native AR recovery')
    p.add_argument('--upstream-repo', required=True, type=Path, help='RynnVLA-002 checkout at pinned worldvla commit')
    p.add_argument('--model-root', required=True, type=Path, help='HF WorldVLA root containing base_model, chameleon/tokenizer, model_256')
    p.add_argument('--model-path', type=Path, help='optional suite target model directory override')
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--checkpoint', type=Path, help='rich draft checkpoint; defaults to weights/worldvla/<suite>.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=indices, default=indices('0-9'))
    p.add_argument('--initials', type=indices, default=indices('0-4'))
    p.add_argument('--mode', choices=('native', 'spec', 'both'), default='both')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--xyz-threshold', type=float, default=.01, help='controller goal delta, metres')
    p.add_argument('--rotation-threshold', type=float, default=.05, help='axis-angle controller goal delta component, radians')
    return p

def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row)+'\n')


def rollout(rt, draft, task, initial, mode, output, gate):
    import numpy as np
    from .runtime import environment, image
    from .contract import decode_step, valid, NativeProtocolError
    from .engine import query
    import torch
    env, instruction, obs = environment(rt.suite, task, initial)
    frames = []; rows = []; steps = 0; success = False; failure = None
    initial_hash = hashlib.sha256(np.ascontiguousarray(obs['agentview_image']).tobytes()+np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
    budget = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)[rt.suite]
    try:
        while steps < budget:
            rt.reset(); rt.current_obs = obs; stats = {}; tokens = None
            start = rt.clock()
            # Unmodified canonical target: agentview only, previous/current frames.
            prompt = rt.processor.prompt(frames[-1:]+[image(obs)], instruction)
            image_end = rt.clock()
            try:
                if mode == 'native':
                    begin = rt.clock(); tokens, _ = rt.native(prompt); rt.times['native_s'] = rt.clock()-begin
                else:
                    tokens, stats = query(rt, prompt, draft, gate)
                    assert rt.counts['branch_calls'] == [1, 1, 1] and rt.counts['block_calls'] == 1 and rt.counts['prefill_calls'] == 1
                blocks = valid(tokens, rt.horizon)
            except (ValueError, NativeProtocolError) as e:
                failure = repr(e)
                torch.save(dict(prompt=prompt.cpu(), tokens=tokens, error=failure), output/f'protocol_{task}_{initial}_{mode}_{len(rows)}.pt')
            end = rt.clock()
            row = dict(suite=rt.suite, task=task, init=initial, mode=mode, query=len(rows), policy_s=end-start,
                       image_prompt_s=image_end-start, counts=rt.counts.copy(), timing=rt.times.copy(), failure=failure, output_tokens=tokens, **stats)
            rows.append(row); append(output/'queries.jsonl', row)
            if failure:
                break
            for block in blocks:
                prev = image(obs); obs, _, done, _ = env.step(decode_step(block)); frames.append(prev); steps += 1
                if done:
                    success = True; break
                if steps >= budget:
                    break
            if success:
                break
        return dict(suite=rt.suite, task=task, init=initial, mode=mode, success=success, steps=steps,
                    protocol_failure=failure, initial_hash=initial_hash, queries=len(rows), total_policy_s=sum(r['policy_s'] for r in rows))
    finally:
        env.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if any(t >= 10 for t in args.tasks):
        raise SystemExit('All supported suites have task IDs 0..9')
    if any(i >= 50 for i in args.initials):
        raise SystemExit('LIBERO reset initial IDs must be 0..49')
    # Avoid accidental append/mixing of independent benchmark runs.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit('--output must be an empty directory')
    import torch
    from .runtime import Runtime
    from .rich import RichDraft
    from .engine import AbsoluteGate, DEFAULT_GATE
    gate_config = dict(DEFAULT_GATE, xyz_threshold=args.xyz_threshold, rotation_threshold=args.rotation_threshold)
    gate = AbsoluteGate(gate_config)
    checkpoint = args.checkpoint or Path(__file__).resolve().parents[1]/'weights/worldvla'/f'{args.suite}.pt'
    rt = Runtime(args.suite, args.upstream_repo, args.model_root, args.model_path, args.device)
    draft = None; handles = []
    if args.mode != 'native':
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['horizon'] != rt.horizon:
            raise ValueError('suite/checkpoint horizon mismatch')
        model = RichDraft(ck['horizon']).to(rt.device).eval(); model.load_state_dict(ck['state'])
        for i, head in enumerate(model.heads):
            handles.append(head.register_forward_hook(lambda m, a, o, i=i: rt.counts['branch_calls'].__setitem__(i, rt.counts['branch_calls'][i]+1)))
        class TimedDraft:
            def propose(self, features):
                start = rt.clock(); out = model.propose(features); rt.times['draft_s'] = rt.clock()-start; return out
        draft = TimedDraft()
    manifest = dict(suite=args.suite, tasks=args.tasks, initials=args.initials, mode=args.mode, gate=gate_config,
                    dtype='float32', prompt='unaltered previous/current agentview; no wrist',
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if draft else None,
                    upstream_repo=str(args.upstream_repo.resolve()), model_root=str(args.model_root.resolve()),
                    model_path=str(args.model_path.resolve()) if args.model_path else None,
                    scope='user-selected evaluation; no success, speedup or independent holdout guarantee')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        with torch.inference_mode():
            for initial in args.initials:
                for task in args.tasks:
                    modes = (['native', 'spec'] if task % 2 == 0 else ['spec', 'native']) if args.mode == 'both' else [args.mode]
                    arms = {mode: rollout(rt, draft, task, initial, mode, args.output, gate) for mode in modes}
                    if args.mode == 'both':
                        assert arms['native']['initial_hash'] == arms['spec']['initial_hash'], 'paired reset state mismatch'
                    row = dict(suite=args.suite, task=task, init=initial, order=modes, **arms)
                    append(args.output/'episodes.jsonl', row)
                    print(json.dumps(row), flush=True)
    finally:
        for handle in handles:
            handle.remove()

if __name__ == '__main__':
    main()
