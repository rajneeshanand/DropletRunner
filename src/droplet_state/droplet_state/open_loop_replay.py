#!/usr/bin/env python3

#Open-Loop Replay: Plays back recorded motor commands from a successful closed-loop episode WITHOUT using camera feedback to choose actions.



import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import pandas as pd
import numpy as np
import os
import time
import csv
from datetime import datetime


class OpenLoopReplay(Node):
    def __init__(self):
        super().__init__('open_loop_replay')

        #declare parameter for CSV file path
        self.declare_parameter('csv_file', '')
        csv_file = self.get_parameter('csv_file').get_parameter_value().string_value

        if not csv_file or not os.path.exists(csv_file):
            self.get_logger().error(f"CSV file not found: '{csv_file}'")
            self.get_logger().error("Usage: ros2 run droplet_state open_loop_replay "
                                    "--ros-args -p csv_file:=/path/to/episode.csv")
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

  
        self.df = pd.read_csv(csv_file)               #load the recorded episode
        self.total_steps = len(self.df)
        self.get_logger().info(f"Loaded {self.total_steps} steps from {csv_file}")
        self.get_logger().info(f"Original episode: progress={self.df['progress'].iloc[-1]:.1f}mm")

        self.motor_pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)

        self.current_pos = None
        self.pos_sub = self.create_subscription(
            Float64MultiArray,
            '/droplet_state',
            self.state_callback,
            10
        )

        self.step = 0
        self.running = False
        self.recording = []  #store actual trajectory for comparison

        self.dt = 0.05
        self.timer = None

        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("OPEN-LOOP REPLAY")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Motor commands are PRE-RECORDED from a successful episode.")
        self.get_logger().info("The camera is NOT used to choose actions.")
        self.get_logger().info("This should FAIL — demonstrating closed-loop necessity.")
        self.get_logger().info("=" * 60)
        self.get_logger().info("")
        self.get_logger().info("Place droplet at START position, then press ENTER...")


        import threading
        self.input_thread = threading.Thread(target=self.wait_for_start, daemon=True)
        self.input_thread.start()

    def state_callback(self, msg):
        if len(msg.data) >= 2:
            self.current_pos = (msg.data[0], msg.data[1])

    def wait_for_start(self):
        input()
        self.get_logger().info("Starting open-loop replay in 3 seconds...")
        time.sleep(3)
        self.running = True
        self.step = 0
        self.recording = []
        self.timer = self.create_timer(self.dt, self.replay_step)
        self.get_logger().info("REPLAYING recorded actions (open-loop)...")

    def replay_step(self):
        if not self.running:
            return

        if self.step >= self.total_steps:
            self.finish_episode()
            return

        #get pre-recorded motor commands
        row = self.df.iloc[self.step]
        current1 = float(row['current1'])
        current2 = float(row['current2'])

        msg = Float32MultiArray()
        msg.data = [current1, current2]
        self.motor_pub.publish(msg)

        # Record actual position
        if self.current_pos is not None:
            self.recording.append({
                'step': self.step,
                'x_mm': self.current_pos[0],
                'y_mm': self.current_pos[1],
                'current1': current1,
                'current2': current2,
                'original_x': float(row['x_mm']),
                'original_y': float(row['y_mm']),
                'original_progress': float(row['progress']),
            })

        # Log every 20 steps
        if self.step % 20 == 0:
            pos_str = f"({self.current_pos[0]:.1f},{self.current_pos[1]:.1f})" if self.current_pos else "(?,?)"
            orig_str = f"({row['x_mm']:.1f},{row['y_mm']:.1f})"
            self.get_logger().info(
                f"Step {self.step}: actual={pos_str} original={orig_str} "
                f"cmd=({current1:.1f},{current2:.1f})"
            )

        self.step += 1

    def finish_episode(self):
        self.running = False
        if self.timer:
            self.timer.cancel()

        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.motor_pub.publish(msg)

        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"OPEN-LOOP REPLAY COMPLETE: {self.step} steps")

        #save recording
        if self.recording:
            timestamp = datetime.now().strftime("%H%M%S")
            out_file = f"openloop_replay_{timestamp}.csv"
            with open(out_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.recording[0].keys())
                writer.writeheader()
                writer.writerows(self.recording)
            self.get_logger().info(f"Saved actual trajectory to: {out_file}")

            # Compute drift
            last = self.recording[-1]
            final_actual = (last['x_mm'], last['y_mm'])
            final_original = (last['original_x'], last['original_y'])
            drift = np.sqrt(
                (final_actual[0] - final_original[0])**2 +
                (final_actual[1] - final_original[1])**2
            )
            self.get_logger().info(f"Final actual position:   ({final_actual[0]:.1f}, {final_actual[1]:.1f})")
            self.get_logger().info(f"Final original position: ({final_original[0]:.1f}, {final_original[1]:.1f})")
            self.get_logger().info(f"Drift from original: {drift:.1f} mm")

        self.get_logger().info("=" * 60)
        self.get_logger().info("")
        self.get_logger().info("Press Ctrl+C to exit.")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OpenLoopReplay()
        rclpy.spin(node)
    except FileNotFoundError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()