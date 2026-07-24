#!/usr/bin/env python3

#Dynamixel motor driver for DropletRunner | Receives current (torque) commands and sends to motors 


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
import struct

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102

DXL_ID_1 = 1
DXL_ID_2 = 2

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

class MotorDriver(Node):
    def __init__(self):
        super().__init__('motor_driver')

        self.declare_parameter('port', '/dev/ttyUSB0')
        port_name = self.get_parameter('port').value

        self.port_handler = PortHandler(port_name)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            self.get_logger().error(f'Failed to open port {port_name}')
            raise RuntimeError('Port open failed')

        if not self.port_handler.setBaudRate(BAUDRATE):
            self.get_logger().error('Failed to set baud rate')
            raise RuntimeError('Baud rate failed')

        self.get_logger().info('Rebooting motors to clear errors...')
        self.packet_handler.reboot(self.port_handler, DXL_ID_1)
        self.packet_handler.reboot(self.port_handler, DXL_ID_2)
        import time
        time.sleep(1.0)

        for dxl_id in [DXL_ID_1, DXL_ID_2]:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_OPERATING_MODE, 0)
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
            mode, _, _ = self.packet_handler.read1ByteTxRx(
                self.port_handler, dxl_id, ADDR_OPERATING_MODE)
            self.get_logger().info(
                f'Motor {dxl_id}: mode={mode} (0=current control)')

        self.subscription = self.create_subscription(
            Float32MultiArray, '/motor_commands',
            self.command_callback, 10)

        self.get_logger().info('Motor driver ready (current control mode)')

    def command_callback(self, msg):
        if len(msg.data) < 2:
            return

        cur_1 = int(max(-1750, min(1750, msg.data[0])))
        cur_2 = int(max(-1750, min(1750, msg.data[1])))

        val_1 = struct.unpack('<H', struct.pack('<h', cur_1))[0]
        val_2 = struct.unpack('<H', struct.pack('<h', cur_2))[0]

        self.packet_handler.write2ByteTxRx(
            self.port_handler, DXL_ID_1, ADDR_GOAL_CURRENT, val_1)
        self.packet_handler.write2ByteTxRx(
            self.port_handler, DXL_ID_2, ADDR_GOAL_CURRENT, val_2)

    def destroy_node(self):
        for dxl_id in [DXL_ID_1, DXL_ID_2]:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.port_handler.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()