#!/usr/bin/env python3

#Data collector node for DropletRunner.
#Uses trained DreamerV3 policy with exploration noise to collect episodes. Falls back to random policy if no checkpoint is found.


import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import warnings
warnings.filterwarnings('ignore')

dreamer_path = os.path.expanduser('~/cyberrunner/dreamerv3')  #DreamerV3 to path
sys.path.insert(0, dreamer_path)
sys.path.insert(0, os.path.join(dreamer_path, 'dreamerv3'))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.bridge = CvBridge()

        from droplet_state.path_tracker import PathTracker    #path tracker
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'config', 'path_waypoints_mm.json'),
            os.path.expanduser('~/droplet_runner_ws/src/droplet_state/config/path_waypoints_mm.json'),
        ]
        waypoints_file = None
        for p in possible_paths:
            if os.path.exists(p):
                waypoints_file = p
                break
        self.path_tracker = PathTracker(waypoints_file)

        self.use_learned_policy = False                 #Try to load trained policy, if fails, fall back to random policy
        self.agent = None
        self.agent_state = None
        policy_dir = os.path.expanduser('~/droplet_runner_ws/policy')
        checkpoint_path = os.path.join(policy_dir, 'checkpoint.ckpt')

        if os.path.exists(checkpoint_path):
            try:
                import embodied
                from dreamerv3 import agent as agt

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

                config = embodied.Config(agt.Agent.configs['defaults'])
                config = config.update(agt.Agent.configs['small'])
                config = config.update({
                    'logdir': policy_dir,
                    'task': 'droplet_runner',
                    'batch_size': 1,
                    'batch_length': 1,
                    'encoder.cnn_keys': 'image',
                    'encoder.mlp_keys': 'vector',
                    'decoder.cnn_keys': 'image',
                    'decoder.mlp_keys': 'vector',
                    'jax.platform': 'cpu',
                })

                self.step_counter = embodied.Counter()
                self.agent = agt.Agent(obs_space, act_space, self.step_counter, config)

                # Dummy replay for checkpoint loading
                replay_dir = os.path.join(policy_dir, 'replay')
                os.makedirs(replay_dir, exist_ok=True)
                replay = embodied.replay.Uniform(
                    length=config.batch_length,
                    capacity=int(1e4),
                    directory=replay_dir,
                )

                checkpoint = embodied.Checkpoint(checkpoint_path)
                checkpoint.step = self.step_counter
                checkpoint.agent = self.agent
                checkpoint.replay = replay
                checkpoint.load_or_save()

                self.use_learned_policy = True
                self.get_logger().info('Loaded trained policy — using EXPLORE mode (policy + noise)')
            except Exception as e:
                self.get_logger().warn(f'Failed to load policy: {e}')
                self.get_logger().info('Falling back to random policy')
        else:
            self.get_logger().info('No checkpoint found — using random policy')

        #Episode storage
        self.episode_dir = os.path.expanduser('~/droplet_runner_ws/episodes')
        os.makedirs(self.episode_dir, exist_ok=True)
        self.episode_count = len([
            f for f in os.listdir(self.episode_dir) if f.endswith('.npz')])

        #current state from estimator 
        self.current_frame = None
        self.current_state = None
        self.state_stamp = None


        self.ep_images = []
        self.ep_vectors = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_is_first = []
        self.ep_is_terminal = []


        self.in_episode = False
        self.episode_steps = 0
        self.max_episode_steps = 600
        self.control_hz = 20.0
        self.prev_action = np.array([0.0, 0.0])

        #Hold-action during prediction gaps
        self.consecutive_predict_frames = 0
        self.HOLD_THRESHOLD = 5        # after 5 predict frames, freeze action
        self.last_good_action = np.array([0.0, 0.0])
        self.last_good_pos = np.array([0.0, 0.0])  # last detected position

        self.max_action = 150.0

        #board ROI for patch extraction
        self.roi_x1, self.roi_y1 = 338, 126
        self.roi_x2, self.roi_y2 = 908, 648
        self.board_w_mm = 266.0
        self.board_h_mm = 241.0

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            Float32MultiArray, '/droplet_state',
            self.state_callback, 10)

        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/motor_commands', 10)

        self.control_timer = self.create_timer(
            1.0 / self.control_hz, self.control_step)

        mode_str = "LEARNED POLICY (explore)" if self.use_learned_policy else "RANDOM"
        self.get_logger().info(
            f'Data collector ready [{mode_str}]. Episodes → {self.episode_dir}')
        self.get_logger().info(
            'Place droplet at start, then press ENTER to begin episode.')

        import threading
        self.input_thread = threading.Thread(
            target=self._wait_for_input, daemon=True)
        self.input_thread.start()


    def image_callback(self, msg):
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def state_callback(self, msg):
        self.current_state = np.array(msg.data)
        self.state_stamp = time.time()


    def control_step(self):
        if not self.in_episode:
            self._send_action(np.array([0.0, 0.0]))
            return

        if self.current_state is None or self.current_frame is None:
            return
        if self.state_stamp is None:
            return
        if time.time() - self.state_stamp > 0.5:
            self.get_logger().warn('State data stale, skipping step')
            return

        x_mm, y_mm, alpha, beta, detected = self.current_state

        if detected == 1.0:
            # Real detection — reset predict counter, save good state
            self.consecutive_predict_frames = 0
            self.last_good_pos = np.array([x_mm, y_mm])
            holding = False
        elif detected >= 0.5:
            # Kalman prediction
            self.consecutive_predict_frames += 1
            holding = self.consecutive_predict_frames > self.HOLD_THRESHOLD
        else:
            # Truly lost
            self.consecutive_predict_frames += 1
            holding = True

        patch = self._extract_patch_from_mm(x_mm, y_mm, detected)

        if detected == 1.0:
            #use actual position
            track_result = self.path_tracker.update([x_mm, y_mm])
        elif detected >= 0.5 and not holding:
            # Short Kalman prediction (< 5 frames) — trust it
            track_result = self.path_tracker.update([x_mm, y_mm])
        elif detected >= 0.5 and holding:
            # Long Kalman prediction — use last good position and to avoid path tracker seeing drifted predictions
            track_result = self.path_tracker.update(
                self.last_good_pos.tolist())
        else:
            # Truly lost
            track_result = self.path_tracker.update([0.0, 0.0])

        reward = track_result['reward']
        done = track_result['done']

        #build observation vector
        lookahead = track_result['lookahead_points'].flatten()
        obs_vector = np.array([
            x_mm, y_mm, alpha, beta, *lookahead
        ], dtype=np.float32)

        #choose action
        if holding:
            # HOLD: freeze motor at last good action
            action = self.last_good_action.copy()
        elif self.use_learned_policy:
            action = self._learned_policy(patch, obs_vector, reward)
            self.last_good_action = action.copy()
        else:
            action = self._random_policy()
            self.last_good_action = action.copy()

        #Store transition
        is_first = (self.episode_steps == 0)
        self.ep_images.append(patch)
        self.ep_vectors.append(obs_vector)
        self.ep_actions.append(action.copy())
        self.ep_rewards.append(float(reward))
        self.ep_is_first.append(is_first)
        self.ep_is_terminal.append(done)

        self._send_action(action)                   #send action to motors
        self.prev_action = action.copy()
        self.episode_steps += 1

        if self.episode_steps % 20 == 0:
            mode_tag = "P" if self.use_learned_policy else "R"
            hold_tag = " HOLD" if holding else ""
            self.get_logger().info(
                f'[{mode_tag}] Step {self.episode_steps}: '
                f'pos=({x_mm:.1f},{y_mm:.1f})mm '
                f'progress={track_result["progress"]:.1f}mm '
                f'reward={reward:.3f}{hold_tag}')


        #termination check
        if detected == 0.0 and self.episode_steps > 10:
            done = True
            self.get_logger().info('Episode ended: droplet truly lost')
        if done or self.episode_steps >= self.max_episode_steps:
            if not done:
                self.ep_is_terminal[-1] = True
            self._end_episode()

    #Action policies

    def _learned_policy(self, patch, obs_vector, reward):    #Use trained DreamerV3 policy with exploration noise. mode='explore' adds noise
    
        obs = {
            'image': patch[None],
            'vector': obs_vector[None],
            'reward': np.array([reward], dtype=np.float32),
            'is_first': np.array([self.episode_steps == 0]),
            'is_last': np.array([False]),
            'is_terminal': np.array([False]),
        }

        act, self.agent_state = self.agent.policy(
            obs, self.agent_state, mode='explore')

        # Scale from [-1, 1] to current units
        action_normalized = np.array(act['action'][0])
        action = action_normalized * self.max_action
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def _random_policy(self):          #Random action with smoothing
        noise = np.random.randn(2) * 80.0
        action = 0.7 * self.prev_action + 0.3 * noise
        action = np.clip(action, -self.max_action, self.max_action)
        return action


    def _extract_patch_from_mm(self, x_mm, y_mm, detected):
        if self.current_frame is None or detected < 0.5:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        px = int(self.roi_x1 + (x_mm + self.board_w_mm/2) /
                 self.board_w_mm * (self.roi_x2 - self.roi_x1))
        py = int(self.roi_y1 + (y_mm + self.board_h_mm/2) /
                 self.board_h_mm * (self.roi_y2 - self.roi_y1))

        h, w = self.current_frame.shape[:2]
        half = 32
        px = max(half, min(w - half, px))
        py = max(half, min(h - half, py))

        patch = self.current_frame[py-half:py+half, px-half:px+half]

        if patch.shape != (64, 64, 3):
            patch = np.zeros((64, 64, 3), dtype=np.uint8)

        return patch.astype(np.uint8)

    def _send_action(self, action):
        msg = Float32MultiArray()
        msg.data = [float(action[0]), float(action[1])]
        self.motor_pub.publish(msg)

    def _end_episode(self):
        self.in_episode = False

        if len(self.ep_images) == 0:
            self.get_logger().warn('Empty episode, not saving')
            return

        filename = f'episode_{self.episode_count:04d}.npz'
        filepath = os.path.join(self.episode_dir, filename)

        np.savez_compressed(filepath,
            image=np.array(self.ep_images, dtype=np.uint8),
            vector=np.array(self.ep_vectors, dtype=np.float32),
            action=np.array(self.ep_actions, dtype=np.float32),
            reward=np.array(self.ep_rewards, dtype=np.float32),
            is_first=np.array(self.ep_is_first, dtype=bool),
            is_terminal=np.array(self.ep_is_terminal, dtype=bool),
        )

        n_steps = len(self.ep_images)
        total_reward = sum(self.ep_rewards)
        mode_str = "LEARNED" if self.use_learned_policy else "RANDOM"
        print(f'Saved {filename}: {n_steps} steps, reward={total_reward:.1f} [{mode_str}]')

        self.episode_count += 1
        self._reset_buffers()

        try:
            self._send_action(np.array([0.0, 0.0]))
        except:
            pass

        print('Place droplet at start, then press ENTER for next episode.')

    def _reset_buffers(self):
        self.ep_images = []
        self.ep_vectors = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_is_first = []
        self.ep_is_terminal = []
        self.episode_steps = 0
        self.prev_action = np.array([0.0, 0.0])
        self.agent_state = None
        self.path_tracker.reset()
        self.consecutive_predict_frames = 0
        self.last_good_action = np.array([0.0, 0.0])
        self.last_good_pos = np.array([0.0, 0.0])

    def _wait_for_input(self):
        while True:
            input("\n>>> Press ENTER to start a new episode...")
            if not self.in_episode:
                self.in_episode = True
                self._reset_buffers()
                self.get_logger().info('Episode started!')
            else:
                self.get_logger().info('Episode already running.')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if node.in_episode and len(node.ep_images) > 0:
        node.ep_is_terminal[-1] = True
        filename = f'episode_{node.episode_count:04d}.npz'
        filepath = os.path.join(node.episode_dir, filename)
        np.savez_compressed(filepath,
            image=np.array(node.ep_images, dtype=np.uint8),
            vector=np.array(node.ep_vectors, dtype=np.float32),
            action=np.array(node.ep_actions, dtype=np.float32),
            reward=np.array(node.ep_rewards, dtype=np.float32),
            is_first=np.array(node.ep_is_first, dtype=bool),
            is_terminal=np.array(node.ep_is_terminal, dtype=bool),
        )
        print(f'Saved partial {filename}: {len(node.ep_images)} steps')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()