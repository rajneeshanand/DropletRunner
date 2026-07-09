#!/usr/bin/env python3

#Camera publisher node for DropletRunner.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CamPublisher(Node):
    def __init__(self):
        super().__init__('cam_publisher')

        #Camera device index may need adjustment once camera cable is unplugged from control unit (PC)
        
        self.declare_parameter('device_id', 2)
        self.declare_parameter('width', 1200)
        self.declare_parameter('height', 800)
        self.declare_parameter('fps', 15)

        device_id = self.get_parameter('device_id').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
 
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)                                  #camera opens
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera device {device_id}')
            raise RuntimeError('Camera not found')

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps'
        )

        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()

        timer_period = 1.0 / fps                                                                #timer
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()