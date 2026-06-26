import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument, 
    ExecuteProcess, 
    TimerAction, 
    IncludeLaunchDescription
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, 
    PythonExpression,
    PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    navila_package = 'navila_ros2_bridge'

    # -------------------------------------------------------------------------------
    # Launch Configurations and Expressions
    enable_safety = LaunchConfiguration("enable_safety")
    env =           LaunchConfiguration("env")  # "sim" or "lab"

    action_out_topic = PythonExpression(["'/cmd_vel_raw' if '", enable_safety, "' == 'true' else '/cmd_vel'"])
    config_filename = PythonExpression(
        ["'sim_navila_config.yaml' if '", env,
         "' == 'sim' else 'lab_navila_config.yaml'"]
    )
    config_file = PathJoinSubstitution(
        [FindPackageShare(navila_package), 'config', config_filename]
    )
    use_sim_time = PythonExpression(
        ["'true' if '", env, "' == 'sim' else 'false'"]
    )


    return LaunchDescription([
    # -------------------------------------------------------------------------------
    # Declare Launch Arguments and Include Other Launch Files
       DeclareLaunchArgument(
            "enable_safety",
            default_value="true",
            choices=["true", "false"],
            description="If true, start safety_layer_node between action node and twist_mux; if false, action node publishes straight to /cmd_vel.",
        ),
        DeclareLaunchArgument(
            'env',
            default_value='lab',
            choices=['sim', 'lab'],
            description='Environment configuration'
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('husky_navigation'),
                    'launch',
                    'pointcloud_to_laserscan.launch.py'
                )
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                # 'pcd_input': '/sensors/lidar3d_0/points',
                # 'scan_output': 'scan',
                # 'lidar_frame': 'lidar3d_0_laser',
            }.items()
        ),
    # -------------------------------------------------------------------------------
    # ROS2 Nodes
        # Action node: everything from config, except the dynamic output topic.
        Node(
            package=navila_package,
            executable='action_node',
            name='action_node',
            namespace='',
            output='screen',
            emulate_tty=True,
            parameters=[
                config_file,
                {"cmd_vel_topic": action_out_topic},
            ],
        ),

        # Safety layer: fully from config. Started only if enable_safety == true.
        Node(
            package=navila_package,
            executable='safety_layer_node',
            name='safety_layer_node',
            namespace='',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(enable_safety),
            parameters=[config_file],
        ),

        ExecuteProcess(
            cmd=[
                'xterm', '-title', 'NaVILA Goal Input', '-e',
                'bash -c "source /opt/ros/humble/setup.bash && '
                'source /home/ros_ws/install/setup.bash && '
                'ros2 run navila_ros2_bridge instruction_node; '
                'echo DONE; read"'
            ],
            output='screen',
            emulate_tty=True,
        ),

        # NaVILA Node: inference Node
        # Start navila_super_node after 6 secs (params from config). For deterministic ordering you can replace this TimerAction with a RegisterEventHandler(OnProcessStart=...) keyed on a node that signals the sim/pipeline is actually ready.
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package=navila_package,
                    executable='navila_super_node',
                    name='navila_super_node',
                    output='screen',
                    emulate_tty=True,
                    parameters=[config_file],
                ),
            ]
        ),
    ])