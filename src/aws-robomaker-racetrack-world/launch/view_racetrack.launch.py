#    import os
#    from ament_index_python.packages import get_package_share_directory
#    from launch import LaunchDescription
#    from launch.actions import IncludeLaunchDescription
#    from launch.launch_description_sources import PythonLaunchDescriptionSource
#    def generate_launch_description():
#        racetrack_pkg_dir = get_package_share_directory('aws_robomaker_racetrack_world')
#        racetrack_launch_path = os.path.join(warehouse_pkg_dir, 'launch')
#        racetrack_world_cmd = IncludeLaunchDescription(
#            PythonLaunchDescriptionSource([racetrack_launch_path, '/racetrack.launch.py'])
#        )
#        ld = LaunchDescription()
#        ld.add_action(racetrack_world_cmd)
#        return ld

#import os
#from ament_index_python.packages import get_package_share_directory
#from launch import LaunchDescription
#from launch.actions import DeclareLaunchArgument#, ExecuteProcess
#from launch_ros.actions import Node
#from launch.substitutions import LaunchConfiguration
#
#def generate_launch_description():
#    # Get the package directory
#    racetrack_pkg_dir = get_package_share_directory('aws_robomaker_racetrack_world')
#
#    # Path to the world file
#    world_file_path = os.path.join(racetrack_pkg_dir, 'worlds', 'racetrack_day.world')
#
#    # Declare launch arguments
#    gui_arg = DeclareLaunchArgument('gui', default_value='true', description='Show Gazebo GUI')
#    world_version_arg = DeclareLaunchArgument('world_version', default_value='day', description='World version (day or night)')
#
#    # Gazebo server (loads the world file)
#    gzserver = Node(
#        package='gazebo_ros',
#        executable='gzserver',
#        arguments=['--verbose', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_init.so', world_file_path],
#        output='screen'
#    )
##    from launch.actions import ExecuteProcess
#    
##    gzserver = ExecuteProcess(
##        cmd=['ign', 'gazebo', '-v', '4', world_file_path],
##        output='screen'
##    )
#    # Gazebo client (GUI for Gazebo)
#    gzclient = Node(
#        package='gazebo_ros',
#        executable='gzclient',
#        condition=LaunchConfiguration('gui'),  # Uses the 'gui' launch argument
#        output='screen'
#    )
#
#    return LaunchDescription([
#        gui_arg,
#        world_version_arg,
#        gzserver,
#        gzclient,
#    ])


# import os
# import sys

import launch
# import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                [get_package_share_directory(
                    'aws_robomaker_racetrack_world'), '/launch/racetrack.launch.py']
            ),
            launch_arguments={
                'gui': 'true',
                'mode': 'day'
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
