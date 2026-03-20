#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from aerogripper_msgs.msg import ControllerState
import math

class ControllerStateSubscriber(Node):
    def __init__(self):
        super().__init__('controller_state_subscriber')
        
        # 创建订阅者
        self.subscription = self.create_subscription(
            ControllerState,
            '/aerogripper/controller_state',
            self.listener_callback,
            10)
        
        self.get_logger().info('ControllerState 订阅者已启动')
        self.get_logger().info('正在监听 /aerogripper/controller_state 话题...')

    def listener_callback(self, msg):
        """处理接收到的ControllerState消息"""
        
        # 基本信息
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'时间步长 dt: {msg.dt:.6f} 秒')
        
        # 期望值
        self.get_logger().info('期望值:')
        self.get_logger().info(f'  加速度: [{msg.desired_acceleration.x:.4f}, {msg.desired_acceleration.y:.4f}, {msg.desired_acceleration.z:.4f}] m/s²')
        self.get_logger().info(f'  角速度: [{msg.desired_angular_velocity.x:.4f}, {msg.desired_angular_velocity.y:.4f}, {msg.desired_angular_velocity.z:.4f}] rad/s')
        self.get_logger().info(f'  推力:   [{msg.desired_thrust.x:.4f}, {msg.desired_thrust.y:.4f}, {msg.desired_thrust.z:.4f}] N')
        self.get_logger().info(f'  力矩:   [{msg.desired_torque.x:.4f}, {msg.desired_torque.y:.4f}, {msg.desired_torque.z:.4f}] N⋅m')
        
        # 当前状态
        self.get_logger().info('当前状态:')
        self.get_logger().info(f'  位置: [{msg.current_position.x:.4f}, {msg.current_position.y:.4f}, {msg.current_position.z:.4f}] m')
        self.get_logger().info(f'  速度: [{msg.current_velocity.x:.4f}, {msg.current_velocity.y:.4f}, {msg.current_velocity.z:.4f}] m/s')
        
        # 姿态信息（四元数和欧拉角）
        self.get_logger().info('  姿态四元数: [w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}]'.format(
            msg.current_orientation.w, msg.current_orientation.x, 
            msg.current_orientation.y, msg.current_orientation.z))
        
        # 欧拉角（转换为度）
        roll_deg = math.degrees(msg.current_euler_angles.x)
        pitch_deg = math.degrees(msg.current_euler_angles.y)
        yaw_deg = math.degrees(msg.current_euler_angles.z)
        
        self.get_logger().info('  姿态欧拉角:')
        self.get_logger().info(f'    Roll (横滚): {roll_deg:.2f}° ({msg.current_euler_angles.x:.4f} rad)')
        self.get_logger().info(f'    Pitch (俯仰): {pitch_deg:.2f}° ({msg.current_euler_angles.y:.4f} rad)')
        self.get_logger().info(f'    Yaw (偏航): {yaw_deg:.2f}° ({msg.current_euler_angles.z:.4f} rad)')
        
        self.get_logger().info(f'  角速度: [{msg.current_angular_velocity.x:.4f}, {msg.current_angular_velocity.y:.4f}, {msg.current_angular_velocity.z:.4f}] rad/s')
        
        # 参考值
        self.get_logger().info('参考值:')
        self.get_logger().info(f'  位置: [{msg.reference_position.x:.4f}, {msg.reference_position.y:.4f}, {msg.reference_position.z:.4f}] m')
        self.get_logger().info(f'  速度: [{msg.reference_velocity.x:.4f}, {msg.reference_velocity.y:.4f}, {msg.reference_velocity.z:.4f}] m/s')
        
        # 参考姿态欧拉角
        ref_roll_deg = math.degrees(msg.reference_euler_angles.x)
        ref_pitch_deg = math.degrees(msg.reference_euler_angles.y)
        ref_yaw_deg = math.degrees(msg.reference_euler_angles.z)
        
        self.get_logger().info('  姿态欧拉角:')
        self.get_logger().info(f'    Roll (横滚): {ref_roll_deg:.2f}° ({msg.reference_euler_angles.x:.4f} rad)')
        self.get_logger().info(f'    Pitch (俯仰): {ref_pitch_deg:.2f}° ({msg.reference_euler_angles.y:.4f} rad)')
        self.get_logger().info(f'    Yaw (偏航): {ref_yaw_deg:.2f}° ({msg.reference_euler_angles.z:.4f} rad)')
        
        # 输出值
        self.get_logger().info('输出值:')
        self.get_logger().info(f'  油门: [{msg.throttle_output[0]:.4f}, {msg.throttle_output[1]:.4f}, {msg.throttle_output[2]:.4f}, {msg.throttle_output[3]:.4f}]')
        
        # 观测值
        self.get_logger().info('观测值:')
        self.get_logger().info(f'  绳子拉力: [{msg.tether_force_bodyframe.x:.4f}, {msg.tether_force_bodyframe.y:.4f}, {msg.tether_force_bodyframe.z:.4f}] N')
        
        # 姿态误差（欧拉角差值）
        roll_error = msg.reference_euler_angles.x - msg.current_euler_angles.x
        pitch_error = msg.reference_euler_angles.y - msg.current_euler_angles.y
        yaw_error = msg.reference_euler_angles.z - msg.current_euler_angles.z
        
        # 处理偏航角误差的周期性
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
            
        self.get_logger().info('姿态误差 (参考 - 当前):')
        self.get_logger().info(f'  Roll误差: {math.degrees(roll_error):.2f}° ({roll_error:.4f} rad)')
        self.get_logger().info(f'  Pitch误差: {math.degrees(pitch_error):.2f}° ({pitch_error:.4f} rad)')
        self.get_logger().info(f'  Yaw误差: {math.degrees(yaw_error):.2f}° ({yaw_error:.4f} rad)')

def main(args=None):
    rclpy.init(args=args)
    
    subscriber = ControllerStateSubscriber()
    
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
