#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import json
import time
import os
import datetime

from droplet_state.path_tracker import PathTracker


class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller')

        # process gains (tuned for DropletRunner)
        self.K_p = 4.0
        self.K_d = 2.0
        self.K_i = 0.1

        #anti-windup clamping limit the integral accumulator
        self.max_integral = 25.0

        self.sign_m1 = 1.0    # Sign for Motor 1 (Y-axis) choosen by trail and error
        self.sign_m2 = -1.0   # Sign for Motor 2 (X-axis)

        self.max_action = 150.0

        self.waypoint_threshold = 8.0

        self.max_steps = 600   # Max steps per episode 

        self.dt = 0.05  # 20 Hz, same as data_collector

        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/motor_commands', 10)
        self.state_sub = self.create_subscription(
            Float32MultiArray, '/droplet_state', self.state_callback, 10)

        #state variables
        self.current_state = None
        self.prev_error = np.array([0.0, 0.0])
        self.integral_error = np.array([0.0, 0.0])
        self.current_waypoint_idx = 0
        self.step_count = 0
        self.episode_count = 0

        #load waypoints (for PID target tracking)
        self.waypoints = self.load_waypoints()
        self.total_path_length = self.compute_path_length()

        # PathTracker for progress 
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'config', 'path_waypoints_mm.json'),
            os.path.expanduser('~/droplet_runner_ws/src/droplet_state/config/path_waypoints_mm.json'),
        ]
        waypoints_file = None
        for p in possible_paths:
            if os.path.exists(p):
                waypoints_file = p
                break
        if waypoints_file is None:
            self.get_logger().error("Could not find path_waypoints_mm.json for PathTracker!")
            raise FileNotFoundError("path_waypoints_mm.json not found")
        self.path_tracker = PathTracker(waypoints_file)

        # Results storage
        self.results = []
        self.current_episode_data = {
            'positions': [],
            'actions': [],
            'progress': [],
            'timestamps': []
        }

        # Episode state
        self.running = False
        self.start_time = None

        #action logging 
        self.action_log_file = None

        self.get_logger().info("=" * 60)
        self.get_logger().info("PID BASELINE CONTROLLER")
        self.get_logger().info(f"  K_p = {self.K_p}, K_i = {self.K_i}, K_d = {self.K_d}")
        self.get_logger().info(f"  Motor signs: M1={self.sign_m1}, M2={self.sign_m2}")
        self.get_logger().info(f"  Max action: ±{self.max_action}")
        self.get_logger().info(f"  Waypoint threshold: {self.waypoint_threshold} mm")
        self.get_logger().info(f"  Waypoints loaded: {len(self.waypoints)}")
        self.get_logger().info(f"  Total path length: {self.total_path_length:.1f} mm")
        self.get_logger().info(f"  Using PathTracker (same as MBRL) for progress")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Place droplet at START and press Enter...")
        self.control_timer = self.create_timer(self.dt, self.control_loop)


        import threading
        self.input_thread = threading.Thread(target=self.wait_for_input, daemon=True)
        self.input_thread.start()

    def load_waypoints(self):
        possible_paths = [
            os.path.expanduser('~/droplet_runner_ws/src/droplet_state/config/path_waypoints_mm.json'),
            os.path.expanduser('~/droplet_runner_ws/config/path_waypoints_mm.json'),
            os.path.expanduser('~/droplet_runner_ws/path_waypoints_mm.json'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                waypoints = np.array(data)
                self.get_logger().info(f"Loaded {len(waypoints)} waypoints from {path}")
                return waypoints

        self.get_logger().error("Could not find path_waypoints_mm.json!")
        raise FileNotFoundError("path_waypoints_mm.json not found")

    def compute_path_length(self):
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
        return total

    def state_callback(self, msg):
        self.current_state = np.array(msg.data)

    def wait_for_input(self):
        while rclpy.ok():
            input("\n>>> Press ENTER to start a new episode...")
            if not self.running:
                self.start_episode()

    def start_episode(self):
        self.episode_count += 1
        self.step_count = 0
        self.current_waypoint_idx = 0
        self.prev_error = np.array([0.0, 0.0])
        self.integral_error = np.array([0.0, 0.0])
        self.running = True
        self.start_time = time.time()
        self.current_episode_data = {
            'positions': [],
            'actions': [],
            'progress': [],
            'timestamps': []
        }

        self.path_tracker.reset()

        # Open CSV log file 
        ts = datetime.datetime.now().strftime('%H%M%S')
        log_dir = os.path.expanduser('~/droplet_runner_ws/pid_logs')
        os.makedirs(log_dir, exist_ok=True)
        logpath = os.path.join(log_dir, f'pid_actions_ep{ts}.csv')
        self.action_log_file = open(logpath, 'w')
        self.action_log_file.write("step,x_mm,y_mm,action_norm1,action_norm2,current1,current2,progress\n")

        self.get_logger().info(f"\n--- Episode {self.episode_count} STARTED (PID) ---")

    def end_episode(self, success, reason=""):
        self.running = False
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Send zero command
        stop_msg = Float32MultiArray()
        stop_msg.data = [0.0, 0.0]
        self.motor_pub.publish(stop_msg)

        # Close CSV log
        if self.action_log_file is not None:
            self.action_log_file.close()
            self.action_log_file = None

        # Compute final progress
        final_progress = 0.0
        if self.current_episode_data['progress']:
            final_progress = max(self.current_episode_data['progress'])

        result = {
            'episode': self.episode_count,
            'success': success,
            'reason': reason,
            'steps': self.step_count,
            'time_seconds': round(elapsed, 2),
            'max_progress_mm': round(final_progress, 1),
            'progress_percent': round(100.0 * final_progress / self.total_path_length, 1),
            'waypoints_reached': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
        }
        self.results.append(result)

        status = "SUCCESS ✓" if success else "FAILED ✗"
        self.get_logger().info(f"\n--- Episode {self.episode_count} {status} ---")
        self.get_logger().info(f"  Reason: {reason}")
        self.get_logger().info(f"  Steps: {self.step_count}, Time: {elapsed:.1f}s")
        self.get_logger().info(f"  Progress: {final_progress:.1f}mm / {self.total_path_length:.1f}mm "
                              f"({result['progress_percent']}%)")
        self.get_logger().info(f"  Waypoints: {self.current_waypoint_idx}/{len(self.waypoints)}")

        # Save results after each episode
        self.save_results()

        # Print running statistics
        successes = sum(1 for r in self.results if r['success'])
        self.get_logger().info(f"\n  Running total: {successes}/{len(self.results)} success "
                              f"({100*successes/len(self.results):.0f}%)")
        self.get_logger().info(f"\nPlace droplet at START and press Enter for next episode...")

    def save_results(self):
        output_path = os.path.expanduser('~/droplet_runner_ws/pid_results.json')

        if self.results:
            successes = sum(1 for r in self.results if r['success'])
            avg_progress = np.mean([r['max_progress_mm'] for r in self.results])

            output = {
                'controller': 'PID Controller',
                'parameters': {
                    'K_p': self.K_p,
                    'K_d': self.K_d,
                    'K_i': self.K_i,
                    'sign_m1': self.sign_m1,
                    'sign_m2': self.sign_m2,
                    'max_action': self.max_action,
                    'waypoint_threshold': self.waypoint_threshold,
                },
                'summary': {
                    'total_episodes': len(self.results),
                    'successes': successes,
                    'success_rate': round(100 * successes / len(self.results), 1),
                    'avg_progress_mm': round(avg_progress, 1),
                    'avg_progress_percent': round(100 * avg_progress / self.total_path_length, 1),
                },
                'episodes': self.results
            }
        else:
            output = {'episodes': []}

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        self.get_logger().info(f"  Results saved to {output_path}")

    def control_loop(self):
        if not self.running or self.current_state is None:
            return

        self.step_count += 1

        x_mm = self.current_state[0]
        y_mm = self.current_state[1]
        detected = self.current_state[4] if len(self.current_state) > 4 else 1.0
        pos = np.array([x_mm, y_mm])

        # Compute progress using PathTracker
        if detected > 0.5:
            track_result = self.path_tracker.update([x_mm, y_mm])
        else:
            track_result = self.path_tracker.update([0.0, 0.0])
        progress = track_result['progress']
        path_distance = track_result['distance']

        # Record data
        self.current_episode_data['positions'].append([float(x_mm), float(y_mm)])
        self.current_episode_data['progress'].append(float(progress))
        self.current_episode_data['timestamps'].append(time.time() - self.start_time)

        #termination check via PathTracker
        if track_result['done']:
            if track_result['success']:
                self.end_episode(True, f"Reached goal! progress={progress:.1f}mm")
            else:
                self.end_episode(False, f"Deviation too large: {path_distance:.1f}mm")
            return

        if self.step_count >= self.max_steps:
            self.end_episode(False, f"Max steps reached: {self.max_steps}")
            return


        # Find target waypoint: advance if close enough to current target
        while self.current_waypoint_idx < len(self.waypoints) - 1:
            dist_to_current = np.linalg.norm(pos - self.waypoints[self.current_waypoint_idx])
            if dist_to_current < self.waypoint_threshold:
                self.current_waypoint_idx += 1
            else:
                break

        target = self.waypoints[self.current_waypoint_idx]

        #Compute error (target - current)
        error = target - pos  # [error_x, error_y] in mm

        #compute derivative of error
        d_error = (error - self.prev_error) / self.dt
        self.prev_error = error.copy()

        #integral with anti-windup
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)

        #Motor 1 controls error[0], Motor 2 controls error[1]
        motor1_cmd = self.sign_m1 * (self.K_p * error[0] + self.K_i * self.integral_error[0] + self.K_d * d_error[0])
        motor2_cmd = self.sign_m2 * (self.K_p * error[1] + self.K_i * self.integral_error[1] + self.K_d * d_error[1])

        #clip to max action range
        motor1_cmd = np.clip(motor1_cmd, -self.max_action, self.max_action)
        motor2_cmd = np.clip(motor2_cmd, -self.max_action, self.max_action)

        cmd_msg = Float32MultiArray()
        cmd_msg.data = [float(motor1_cmd), float(motor2_cmd)]
        self.motor_pub.publish(cmd_msg)

        self.current_episode_data['actions'].append([float(motor1_cmd), float(motor2_cmd)])

        if self.action_log_file is not None:
            self.action_log_file.write(
                f"{self.step_count},{x_mm:.2f},{y_mm:.2f},"
                f"{motor1_cmd/self.max_action:.4f},{motor2_cmd/self.max_action:.4f},"
                f"{motor1_cmd:.1f},{motor2_cmd:.1f},"
                f"{progress:.1f}\n")

        # Log every 40 steps
        if self.step_count % 40 == 0:
            self.get_logger().info(
                f"  Step {self.step_count:4d} | pos=({x_mm:.1f}, {y_mm:.1f}) | "
                f"target=wp[{self.current_waypoint_idx}]=({target[0]:.1f}, {target[1]:.1f}) | "
                f"err=({error[0]:.1f}, {error[1]:.1f}) | "
                f"cmd=({motor1_cmd:.0f}, {motor2_cmd:.0f}) | "
                f"progress={progress:.1f}mm ({100*progress/self.total_path_length:.0f}%)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nShutting down PID controller...")
        # Send zero command
        stop_msg = Float32MultiArray()
        stop_msg.data = [0.0, 0.0]
        node.motor_pub.publish(stop_msg)
        # Close log file
        if node.action_log_file is not None:
            node.action_log_file.close()
            node.action_log_file = None
        # Save final results
        node.save_results()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()