from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取 config 文件路径
    config = os.path.join(
        get_package_share_directory('aerogripper_simulator'),
        'config',
        'simulator.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='aerogripper_simulator',
            executable='simulator',
            name='aerogripper_simulator',
            parameters=[config],
            output='screen'
        )
    ])