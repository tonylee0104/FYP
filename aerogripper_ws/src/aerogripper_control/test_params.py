#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import GetParameters
import time

class ParamTestNode(Node):
    def __init__(self):
        super().__init__('param_test_node')
        
        # 等待参数服务可用
        time.sleep(2)
        
        # 获取参数
        self.get_logger().info("Testing parameter loading...")
        
        # 测试单个参数
        try:
            mass = self.get_parameter('mass').value
            self.get_logger().info(f"mass: {mass}")
        except Exception as e:
            self.get_logger().error(f"Failed to get mass parameter: {e}")
            
        try:
            thrust_coeff = self.get_parameter('thrust_coeff').value
            self.get_logger().info(f"thrust_coeff: {thrust_coeff}")
        except Exception as e:
            self.get_logger().error(f"Failed to get thrust_coeff parameter: {e}")
            
        try:
            kp_pos = self.get_parameter('kp_pos').value
            self.get_logger().info(f"kp_pos: {kp_pos}")
        except Exception as e:
            self.get_logger().error(f"Failed to get kp_pos parameter: {e}")
            
        # 列出所有参数
        self.get_logger().info("Listing all parameters:")
        param_names = self.list_parameters([], 10)
        for name in param_names.names:
            try:
                param = self.get_parameter(name)
                self.get_logger().info(f"  {name}: {param.value}")
            except Exception as e:
                self.get_logger().error(f"Failed to get {name}: {e}")

def main():
    rclpy.init()
    node = ParamTestNode()
    
    # 运行一段时间后退出
    time.sleep(5)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
