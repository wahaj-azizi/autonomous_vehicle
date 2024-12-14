# # import launch
# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, Log
# from launch.substitutions import LaunchConfiguration

# from launch_ros.actions import Node

# def generate_launch_description():
#     # Define the launch configuration for the DDQN training node
#     return LaunchDescription([
#         DeclareLaunchArgument('vehicle_name', default_value='audibot', description='Name of the vehicle'),
#         DeclareLaunchArgument('learning_rate', default_value='0.001', description='Learning rate for the DDQN agent'),
#         DeclareLaunchArgument('gamma', default_value='0.99', description='Discount factor for future rewards'),
#         DeclareLaunchArgument('epsilon', default_value='1.0', description='Initial exploration rate'),
#         DeclareLaunchArgument('epsilon_min', default_value='0.01', description='Minimum exploration rate'),
#         DeclareLaunchArgument('epsilon_decay', default_value='0.995', description='Decay rate for epsilon'),
#         DeclareLaunchArgument('batch_size', default_value='32', description='Batch size for training'),
#         DeclareLaunchArgument('replay_buffer_size', default_value='10000', description='Size of the replay buffer'),
#         DeclareLaunchArgument('target_update_interval', default_value='100', description='Frequency of target network updates'),
#         DeclareLaunchArgument('num_episodes', default_value='1000', description='Number of training episodes'),

#         # Node to launch the DDQN training script
#         Node(
#             package='ddqn_audibot',
#             executable='train_ddqn',
#             name='train_ddqn_node',
#             output='screen',
#             parameters=[
#                 {'vehicle_name': LaunchConfiguration('vehicle_name')},
#                 {'learning_rate': LaunchConfiguration('learning_rate')},
#                 {'gamma': LaunchConfiguration('gamma')},
#                 {'epsilon': LaunchConfiguration('epsilon')},
#                 {'epsilon_min': LaunchConfiguration('epsilon_min')},
#                 {'epsilon_decay': LaunchConfiguration('epsilon_decay')},
#                 {'batch_size': LaunchConfiguration('batch_size')},
#                 {'replay_buffer_size': LaunchConfiguration('replay_buffer_size')},
#                 {'target_update_interval': LaunchConfiguration('target_update_interval')},
#                 {'num_episodes': LaunchConfiguration('num_episodes')}
#             ],
#             remappings=[
#                 # Add any topic remappings if needed
#             ]
#         ),
        
#         # Logging for confirmation of launch
#         Log(
#             condition=LaunchConfiguration('verbose').perform(),
#             msg="Launching DDQN training for vehicle: " + LaunchConfiguration('vehicle_name')
#         )
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ddqn_audibot',
            executable='ddqn_agent',
            name='ddqn_agent',
            output='screen',
            parameters=[
                {'action_size': 4},
                {'learning_rate': 0.001},
                {'gamma': 0.99},
                {'epsilon': 1.0},
                {'epsilon_decay': 0.995},
                {'min_epsilon': 0.01},
                {'batch_size': 32}
            ]
        )
    ])
