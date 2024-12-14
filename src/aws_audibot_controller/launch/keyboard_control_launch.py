from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aws_audibot_controller',
            executable='keyboard_control',
            name='keyboard_control_node',
            output='screen',
        )
    ])
