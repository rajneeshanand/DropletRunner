#!/usr/bin/env python3
"""
Offline DreamerV3 training for DropletRunner.
This script directly trains from the replay buffer WITHOUT running
an environment (pure offline / batch RL).

Key insight: embodied.run.train() is for ONLINE training.
For OFFLINE, we manually loop: sample batch -> train -> log -> repeat.
"""

import os
import sys
import pathlib
import warnings
import argparse
import glob
import time

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import numpy as np


def convert_episodes(source_dir, replay_dir):
    """Convert episode_XXXX.npz to embodied chunk format."""
    os.makedirs(replay_dir, exist_ok=True)
    
    # Check if already converted
    existing = glob.glob(os.path.join(replay_dir, '*.npz'))
    if existing:
        print(f"Replay dir already has {len(existing)} files, skipping conversion")
        return
    
    source_files = sorted(glob.glob(os.path.join(source_dir, 'episode_*.npz')))
    print(f"Converting {len(source_files)} episodes...")

    for i, src_path in enumerate(source_files):
        data = np.load(src_path)
        n = len(data['reward'])

        actions = data["action"].astype(np.float32) / 150.0
        actions = np.clip(actions, -1.0, 1.0)

        ep = {
            'image': data['image'],
            'vector': data['vector'].astype(np.float32),
            'action': actions,
            'reward': data['reward'].astype(np.float32),
            'is_first': data['is_first'],
            'is_terminal': data['is_terminal'],
            'is_last': data['is_terminal'].copy(),
        }

        # Thomas's chunk format: {time}-{uuid}-{successor}-{length}.npz
        timestamp = f'{i:016d}'
        uuid = f'{i:016d}'
        successor = '0000000000000000'
        filename = f'{timestamp}-{uuid}-{successor}-{n}.npz'
        np.savez(os.path.join(replay_dir, filename), **ep)

    print(f"Done: {len(source_files)} episodes converted\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes_dir', type=str, default='episodes')
    parser.add_argument('--logdir', type=str, default='logdir')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Number of training updates')
    args = parser.parse_args()

    # Add DreamerV3 to path
    dreamer_path = os.path.expanduser('~/cyberrunner/dreamerv3')
    sys.path.insert(0, dreamer_path)
    sys.path.insert(0, os.path.join(dreamer_path, 'dreamerv3'))

    import embodied
    from dreamerv3 import agent as agt

    # Convert episodes
    replay_dir = os.path.join(args.logdir, 'replay')
    convert_episodes(args.episodes_dir, replay_dir)

    # Configure DreamerV3
    print("=" * 60)
    print("Configuring DreamerV3")
    print("=" * 60)

    config = embodied.Config(agt.Agent.configs['defaults'])
    config = config.update(agt.Agent.configs['small'])
    config = config.update({
        'logdir': args.logdir,
        'task': 'droplet_runner',
        'replay': 'uniform',
        'replay_size': int(1e6),
        'replay_online': False,
        'batch_size': 16,
        'batch_length': 64,
        'run.train_ratio': 128,
        'run.log_every': 60,
        'run.save_every': 20,
        'encoder.cnn_keys': 'image',
        'encoder.mlp_keys': 'vector',
        'decoder.cnn_keys': 'image',
        'decoder.mlp_keys': 'vector',
    })

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')

    # Create observation and action spaces manually
    obs_space = {
        'image': embodied.Space(np.uint8, (64, 64, 3)),
        'vector': embodied.Space(np.float32, (14,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    act_space = {
        'action': embodied.Space(np.float32, (2,), -1.0, 1.0),
        'reset': embodied.Space(bool),
    }

    print(f"Obs space: image(64,64,3) + vector(14,)")
    print(f"Act space: action(2,) in [-1,1]")

    # Create replay buffer
    print("\nLoading replay buffer...")
    replay = embodied.replay.Uniform(
        length=config.batch_length,
        capacity=config.replay_size,
        directory=replay_dir,
    )
    print(f"Replay: {len(replay)} steps loaded")

    if len(replay) < config.batch_size * config.batch_length:
        print(f"ERROR: Not enough data. Need at least {config.batch_size * config.batch_length} "
              f"steps, have {len(replay)}")
        sys.exit(1)

    # Create agent
    print("\nCreating agent...")
    step = embodied.Counter()
    agent = agt.Agent(obs_space, act_space, step, config)

    # Set up logger
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    ])

    # Load checkpoint if exists
    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    checkpoint.load_or_save()

    # Create dataset iterator from replay
    dataset = agent.dataset(replay.dataset)

    # ── OFFLINE TRAINING LOOP ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"Starting offline training for {args.steps} updates")
    print(f"Batch size: {config.batch_size}, Batch length: {config.batch_length}")
    print("=" * 60 + "\n")

    state = None
    metrics = embodied.Metrics()
    last_log_time = time.time()
    last_save_step = 0

    for i in range(args.steps):
        # Sample batch and train
        batch = next(dataset)
        outs, state, mets = agent.train(batch, state)
        metrics.add(mets, prefix='train')
        step.increment()

        # Log periodically
        if time.time() - last_log_time >= config.run.log_every:
            agg = metrics.result()
            report = agent.report(batch)
            report = {k: v for k, v in report.items() if 'train/' + k not in agg}
            logger.add(agg)
            logger.add(report, prefix='report')
            logger.add(replay.stats, prefix='replay')
            logger.write(fps=True)
            last_log_time = time.time()
            
            # Print progress
            elapsed = time.time() - start_time if 'start_time' in dir() else 0
            print(f"Step {i+1}/{args.steps} "
                  f"({100*(i+1)/args.steps:.1f}%) "
                  f"[{elapsed/60:.1f} min]")

        # Save checkpoint periodically
        if (i + 1) % (args.steps // 10) == 0 or i == args.steps - 1:
            print(f"Saving checkpoint at step {i+1}...")
            checkpoint.save()

    # Final save
    checkpoint.save()
    logger.write()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoint: {args.logdir}/checkpoint.ckpt")
    print(f"Total steps: {args.steps}")
    print("=" * 60)


start_time = time.time()

if __name__ == '__main__':
    main()
