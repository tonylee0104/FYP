from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包的共享目录路径
    pkg_share = get_package_share_directory('aerogripper_control')
    
    # 构建config.yaml的完整路径
    config_file = os.path.join(pkg_share, 'config', 'config.yaml')
    
    # 打印配置文件路径用于调试
    print(f"Config file path: {config_file}")
    print(f"Config file exists: {os.path.exists(config_file)}")
    
    return LaunchDescription([
        # 启动主节点，它会创建所有三个组件
        Node(
            package='aerogripper_control',
            executable='aerogripper_control_node',
            name='aerogripper_control',
            parameters=[config_file],
            output='screen'
        ),
    ])