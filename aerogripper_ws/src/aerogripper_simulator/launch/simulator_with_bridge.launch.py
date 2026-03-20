from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("aerogripper_simulator"),
        "config",
        "simulator.yaml",
    )
    rviz_config = os.path.join(
        get_package_share_directory("aerogripper_simulator"),
        "config",
        "simulator.rviz",
    )

    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Launch RViz2 with simulator visualization",
        ),
        Node(
            package="aerogripper_simulator",
            executable="simulator",
            name="aerogripper_simulator",
            parameters=[config],
            output="screen",
        ),
        Node(
            package="aerogripper_simulator",
            executable="px4_bridge",
            name="aerogripper_px4_bridge",
            parameters=[
                {
                    "thrust_coeff": 6.503575,
                    "thrust_arm_length": 0.055,
                    "torque_coeff": 0.002,
                }
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="aerogripper_rviz",
            arguments=["-d", rviz_config],
            output="screen",
            condition=IfCondition(use_rviz),
        ),
    ])
