#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8
import sys
import termios
import tty
import select  # To check for keyboard inputs without blocking

class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control')
        self.steering_angle = 0.0  # Radians
        self.throttle = 0.0  # 0 to 1
        self.brake = 0.0  # Nm
        self.gear = 0  # Drive = 0, Reverse = 1

        # Publishers for control commands
        self.steering_pub = self.create_publisher(Float64, '/audibot/steering_cmd', 10)
        self.throttle_pub = self.create_publisher(Float64, '/audibot/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/audibot/brake_cmd', 10)
        self.gear_pub = self.create_publisher(UInt8, '/audibot/gear_cmd', 10)

        # Timer to publish control commands at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_commands)
        self.get_logger().info("Keyboard control node started.\n")

    def publish_commands(self):
        # Publish control values
        self.steering_pub.publish(Float64(data=self.steering_angle))
        self.throttle_pub.publish(Float64(data=self.throttle))
        self.brake_pub.publish(Float64(data=self.brake))
        self.gear_pub.publish(UInt8(data=self.gear))

    def process_key(self, key):
        if key == 'w':  # Increase throttle
            self.throttle = min(self.throttle + 0.01, 1.0)
            self.brake = 0.0
            self.get_logger().info(f"Throttle increased: {self.throttle:.2f}")

        elif key == 's':  # Decrease throttle
            self.throttle = max(self.throttle - 0.1, 0.0)
            self.get_logger().info(f"Throttle decreased: {self.throttle:.2f}")

        elif key == 'x':  # Apply brake
            self.brake = 300.0
            self.throttle = 0.0
            self.get_logger().info("Brake applied.")

        elif key == 'a':  # Turn left
            self.steering_angle = max(self.steering_angle + 0.2, 0.0)
            self.get_logger().info(f"Steering left: {self.steering_angle:.2f}")

        elif key == 'd':  # Turn right
            self.steering_angle = min(self.steering_angle - 0.2, .0)
            self.get_logger().info(f"Steering right: {self.steering_angle:.2f}")

        elif key == 'r':  # Reverse gear
            self.gear = 1
            self.get_logger().info("Gear set to Reverse.")

        elif key == 'f':  # Drive gear
            self.gear = 0
            self.get_logger().info("Gear set to Drive.")

        elif key == 'z':  # Reset steering
            self.steering_angle = 0.0
            self.get_logger().info("Steering reset.")

        elif key == 'q':  # Quit
            self.get_logger().info("Exiting keyboard control.")
            rclpy.shutdown()
            sys.exit(0)

    def run(self):
        # Setup for non-blocking keyboard input
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.get_logger().info(
            "Keyboard Control for Audibot:\n"
            "-----------------------------\n"
            "w: Increase throttle\n"
            "s: Decrease throttle\n"
            "x: Apply brake\n"
            "a: Turn left\n"
            "d: Turn right\n"
            "r: Reverse gear\n"
            "f: Drive gear\n"
            "z: Reset steering\n"
            "q: Quit\n"
        )

        try:
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0)[0]:  # Non-blocking key check
                    key = sys.stdin.read(1)
                    self.process_key(key)
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    try:
        node.run()
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
