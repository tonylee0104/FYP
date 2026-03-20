#!/usr/bin/env python3
"""
Tethered Rigid Body Simulator: 精确物理仿真器
- 支持运动锚点（通过 odom topic 输入位置/速度）
- 支持卷扬机动态收放绳（绳速 + 绳加速度）
- 精确处理约束：包含 d²L/dt² 和锚点加速度
- 所有加速度通过数值微分获得（带低通滤波）
- 完全参数化，符合 ROS2 最佳实践
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Wrench, AccelStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
from aerogripper_simulator.utils import LowPassFilter


class AergripperSimulator(Node):
    def __init__(self):
        super().__init__('aerogripper_simulator')
        
        # === 参数加载（全部来自 simulator.yaml）===
        self._declare_parameters()
        self._load_parameters()
        
        # === 状态初始化 ===
        self._reset_state()
        
        # === 加速度滤波器（用于数值微分）===
        self.anchor_acc_filter = [
            LowPassFilter(self.accel_filter_tau, self.dt),
            LowPassFilter(self.accel_filter_tau, self.dt),
            LowPassFilter(self.accel_filter_tau, self.dt)
        ]
        self.rope_acc_filter = LowPassFilter(self.accel_filter_tau, self.dt)
        
        # === ROS2 接口 ===
        # 输入：机体坐标系推力/力矩
        self.create_subscription(Wrench, 'wrench_body', self.wrench_callback, 10)
        
        # 输入：运动锚点状态（位置 + 速度）
        self.create_subscription(Odometry, 'anchor_odom', self.anchor_odom_callback, 10)
        
        # 输入：卷扬机绳速（正=放绳，负=收绳）
        self.create_subscription(Float64, 'rope_speed', self.rope_speed_callback, 10)
        
        # 输出：完整状态
        self.odom_pub = self.create_publisher(Odometry, 'simulator/odom', 10)
        self.tension_pub = self.create_publisher(Float64, 'simulator/tension', 10)
        self.accel_pub = self.create_publisher(AccelStamped, 'simulator/accel', 10)
        
        # 仿真主循环（固定步长）
        self.create_timer(self.dt, self.simulation_step)
        # 状态发布循环
        self.create_timer(1.0 / self.publish_rate, self.publish_state)
        
        self.get_logger().info('Aerogripper Simulator initialized with precise constraint dynamics')

    def _declare_parameters(self):
        """声明所有参数（支持 launch 文件覆盖）"""
        # 物理参数
        self.declare_parameter('mass', 0.1)
        self.declare_parameter('body_dims', [0.05, 0.05, 0.1])
        self.declare_parameter('gravity', [0.0, 0.0, -9.80665])
        self.declare_parameter('attach_offset', [0.0, 0.0, 0.05])
        self.declare_parameter('inertia', [0.0001, 0.0001, 0.0001])
        
        # 初始状态
        self.declare_parameter('initial_position', [0.0, 0.0, -0.5])
        self.declare_parameter('initial_rope_length', 0.5)
        self.declare_parameter('initial_rope_speed', 0.0)
        
        # 仿真参数
        self.declare_parameter('simulation_dt', 0.001)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('acceleration_filter_tau', 0.01)

        # 数值稳定性参数
        self.declare_parameter('max_linear_accel', 80.0)
        self.declare_parameter('max_angular_accel', 200.0)
        self.declare_parameter('max_linear_speed', 30.0)
        self.declare_parameter('max_angular_speed', 80.0)
        self.declare_parameter('max_wrench_force', 200.0)
        self.declare_parameter('max_wrench_torque', 20.0)
        
        # 锚点默认状态
        self.declare_parameter('anchor_default_position', [0.0, 0.0, 0.0])
        self.declare_parameter('anchor_default_velocity', [0.0, 0.0, 0.0])

    def _load_parameters(self):
        """加载参数到成员变量"""
        # 物理参数
        self.mass = self.get_parameter('mass').value
        self.body_dims = np.array(self.get_parameter('body_dims').value, dtype=np.float64)
        self.gravity = np.array(self.get_parameter('gravity').value, dtype=np.float64)
        self.attach_offset_body = np.array(self.get_parameter('attach_offset').value, dtype=np.float64)
        
        # 初始状态
        self.initial_pos = np.array(self.get_parameter('initial_position').value, dtype=np.float64)
        self.initial_rope_length = self.get_parameter('initial_rope_length').value
        self.initial_rope_speed = self.get_parameter('initial_rope_speed').value
        
        # 仿真参数
        self.dt = self.get_parameter('simulation_dt').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.accel_filter_tau = self.get_parameter('acceleration_filter_tau').value

        # 数值稳定性参数
        self.max_linear_accel = float(self.get_parameter('max_linear_accel').value)
        self.max_angular_accel = float(self.get_parameter('max_angular_accel').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.max_wrench_force = float(self.get_parameter('max_wrench_force').value)
        self.max_wrench_torque = float(self.get_parameter('max_wrench_torque').value)
        
        # 锚点默认状态
        self.anchor_default_pos = np.array(self.get_parameter('anchor_default_position').value, dtype=np.float64)
        self.anchor_default_vel = np.array(self.get_parameter('anchor_default_velocity').value, dtype=np.float64)
        
        # 直接从配置读取惯性
        inertia_values = np.array(self.get_parameter('inertia').value, dtype=np.float64)
        self.inertia = np.diag(inertia_values)
        self.inv_inertia = np.linalg.inv(self.inertia)

    def _compute_cuboid_inertia(self, dims, mass):
        """计算长方体绕质心的惯性张量"""
        a, b, c = dims
        return np.diag([
            (b**2 + c**2) * mass / 12.0,
            (a**2 + c**2) * mass / 12.0,
            (a**2 + b**2) * mass / 12.0
        ])

    def _reset_state(self):
        """重置仿真状态"""
        # 刚体状态
        self.pos = self.initial_pos.copy()  # 世界系质心位置
        self.vel = np.zeros(3)              # 世界系线速度
        self.acc = np.zeros(3)              # 世界系线加速度
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z] 单位四元数
        self.omega = np.zeros(3)            # 机体坐标系角速度
        self.alpha = np.zeros(3)            # 机体坐标系角加速度
        
        # 绳状态
        self.rope_length = self.initial_rope_length
        self.rope_speed = self.initial_rope_speed  # m/s (正=放绳)
        self.rope_speed_prev = self.rope_speed     # 用于微分
        self.rope_acc = 0.0                        # 绳加速度 (m/s²)
        self.tension = 0.0                         # 绳拉力 (N)
        self.rope_taut = False                     # 绳是否绷紧
        
        # 锚点状态（默认静止）
        self.anchor_pos = self.anchor_default_pos.copy()
        self.anchor_vel = self.anchor_default_vel.copy()
        self.anchor_vel_prev = self.anchor_vel.copy()  # 用于微分
        self.anchor_acc = np.zeros(3)                 # 锚点加速度 (m/s²)
        
        # 控制输入
        self.thrust_body = np.zeros(3)
        self.tau_body = np.zeros(3)

    def wrench_callback(self, msg):
        """接收机体坐标系推力/力矩"""
        thrust = np.array([msg.force.x, msg.force.y, msg.force.z], dtype=np.float64)
        torque = np.array([msg.torque.x, msg.torque.y, msg.torque.z], dtype=np.float64)

        # 输入保护：过滤非法值并限幅，避免异常控制量击穿仿真积分
        thrust = np.where(np.isfinite(thrust), thrust, 0.0)
        torque = np.where(np.isfinite(torque), torque, 0.0)
        thrust = np.clip(thrust, -self.max_wrench_force, self.max_wrench_force)
        torque = np.clip(torque, -self.max_wrench_torque, self.max_wrench_torque)

        self.thrust_body = thrust
        self.tau_body = torque

    def _safe_normalize_quat(self, q):
        """稳定归一化四元数；当范数异常时返回单位四元数。"""
        q = np.where(np.isfinite(q), q, 0.0)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    def _state_is_valid(self):
        """检查关键状态是否仍为有限值。"""
        return (
            np.all(np.isfinite(self.pos)) and
            np.all(np.isfinite(self.vel)) and
            np.all(np.isfinite(self.acc)) and
            np.all(np.isfinite(self.omega)) and
            np.all(np.isfinite(self.alpha)) and
            np.all(np.isfinite(self.quat))
        )

    def anchor_odom_callback(self, msg):
        """接收运动锚点状态（位置 + 速度）并计算加速度"""
        # 更新位置/速度
        self.anchor_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ], dtype=np.float64)
        
        new_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ], dtype=np.float64)
        
        # 数值微分计算加速度 + 低通滤波
        for i in range(3):
            raw_acc = (new_vel[i] - self.anchor_vel_prev[i]) / max(self.dt, 1e-9)
            self.anchor_acc[i] = self.anchor_acc_filter[i].update(raw_acc)
        
        self.anchor_vel_prev = self.anchor_vel.copy()
        self.anchor_vel = new_vel

    def rope_speed_callback(self, msg):
        """接收卷扬机绳速并计算绳加速度"""
        new_speed = msg.data
        # 数值微分 + 低通滤波
        raw_rope_acc = (new_speed - self.rope_speed_prev) / max(self.dt, 1e-9)
        self.rope_acc = self.rope_acc_filter.update(raw_rope_acc)
        
        self.rope_speed_prev = self.rope_speed
        self.rope_speed = new_speed

    def simulation_step(self):
        """单步物理仿真（固定步长）"""
        if not self._state_is_valid():
            self.get_logger().warn('Invalid simulator state detected, reset to initial state')
            self._reset_state()
            return

        # 先确保姿态四元数总是有效，避免 Rotation.from_quat 抛异常
        self.quat = self._safe_normalize_quat(self.quat)

        # 1. 更新绳长（卷扬机运动）
        self.rope_length = max(1e-6, self.rope_length + self.rope_speed * self.dt)
        
        # 2. 计算旋转矩阵（四元数 -> 旋转矩阵）
        # scipy 使用 [x, y, z, w] 格式
        rotm = R.from_quat([
            self.quat[1], self.quat[2], self.quat[3], self.quat[0]
        ]).as_matrix()
        
        # 3. 世界系推力
        thrust_world = rotm @ self.thrust_body
        total_force_world = thrust_world + self.mass * self.gravity
        
        # 4. 附着点位置/速度（世界系）
        attach_offset_world = rotm @ self.attach_offset_body
        attach_pos = self.pos + attach_offset_world
        attach_vel = self.vel + np.cross(self.omega, attach_offset_world)
        
        # 5. 应用绳约束（核心：精确处理 d²L/dt² 和 a_anchor）
        self._apply_constraint(
            attach_pos, attach_vel, attach_offset_world,
            total_force_world, rotm
        )

        # 在线性积分前限幅，避免单步速度突变导致发散
        self.acc = np.clip(self.acc, -self.max_linear_accel, self.max_linear_accel)
        
        # 6. 更新线性状态（半隐式欧拉：先速度后位置）
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt
        
        # 7. 更新角状态
        # 计算约束力向量（当绳绷紧时）
        if self.rope_taut and self.tension > 0:
            r_vec = attach_pos - self.anchor_pos
            n = r_vec / (np.linalg.norm(r_vec) + 1e-9)
            constraint_force_world = -self.tension * n  # 指向锚点
        else:
            constraint_force_world = np.zeros(3)
        
        # ===== 旋转动力学：以质心（COM）为参考点 =====
        # 为什么选择质心？
        #   - 重力对质心力矩恒为零（简化方程）
        #   - 惯性张量在质心坐标系中为常数（数值稳定）
        #   - 与IMU物理测量一致（行业标准）
        # 
        # 力矩来源：
        #   1. 输入力矩 tau_body（如螺旋桨推力差）
        #   2. 绳拉力对质心的力矩：r_attach × F_tension
        #      （r_attach = 从质心指向绳连接点的向量）
        #   3. 陀螺力矩：-ω × (I·ω)（已包含在欧拉方程中）
        # 
        # 重力不产生力矩 → 不出现在力矩方程中（正确！）
        total_torque_body = self.tau_body + rotm.T @ np.cross(attach_offset_world, constraint_force_world)
        
        # 欧拉方程：I*alpha = tau - omega × (I*omega)
        gyro_torque = np.cross(self.omega, self.inertia @ self.omega)
        self.alpha = self.inv_inertia @ (total_torque_body - gyro_torque)
        self.alpha = np.clip(self.alpha, -self.max_angular_accel, self.max_angular_accel)
        self.omega += self.alpha * self.dt
        self.omega = np.clip(self.omega, -self.max_angular_speed, self.max_angular_speed)
        
        # 四元数积分（一阶近似 + 归一化）
        omega_quat = np.array([0.0, self.omega[0], self.omega[1], self.omega[2]])
        q = np.array([self.quat[0], self.quat[1], self.quat[2], self.quat[3]])  # [w,x,y,z]
        q_dot = 0.5 * self._quat_multiply(omega_quat, q)
        q += q_dot * self.dt
        q = self._safe_normalize_quat(q)
        self.quat = np.array([q[0], q[1], q[2], q[3]])  # [w,x,y,z]

        # 线性状态限幅，防止瞬时异常导致发散
        self.vel = np.clip(self.vel, -self.max_linear_speed, self.max_linear_speed)

    def _apply_constraint(self, attach_pos, attach_vel, attach_offset_world, total_force_world, rotm):
        """绳约束处理：精确包含 d²L/dt² 和锚点加速度"""
        # 相对向量与距离
        r_vec = attach_pos - self.anchor_pos
        dist = np.linalg.norm(r_vec + 1e-9)
        n = r_vec / dist  # 径向单位向量（从锚点指向附着点）
        
        # 相对速度（考虑锚点运动）
        v_rel = attach_vel - self.anchor_vel
        radial_vel = np.dot(v_rel, n)  # 径向相对速度
        
        # 切向速度平方
        v_t_sq = np.dot(v_rel, v_rel) - radial_vel**2
        
        # === 绳状态机 ===
        if not self.rope_taut:
            # 松弛状态：检查是否应绷紧
            # 条件：距离 >= 绳长 且 径向速度 >= 绳速（即将拉紧）
            if dist >= self.rope_length and radial_vel >= self.rope_speed:
                self.rope_taut = True
                # 冲量修正：强制径向速度 = 绳速 (dL/dt)
                self._apply_impulse_to_match_rope_speed(v_rel, n, attach_offset_world, radial_vel)
            # 松弛时无约束力
            self.tension = 0.0
            self.acc = total_force_world / self.mass
            return
        else:
            # 绷紧状态：强制位置满足约束
            attach_pos = self.anchor_pos + self.rope_length * n
            self.pos = attach_pos - attach_offset_world
            
            # 修正径向速度 = 绳速 (dL/dt)
            v_rel = (self.vel + np.cross(self.omega, attach_offset_world)) - self.anchor_vel
            radial_vel = np.dot(v_rel, n)
            self._apply_impulse_to_match_rope_speed(v_rel, n, attach_offset_world, radial_vel)
            
            # === 精确拉力计算（包含 d²L/dt² 和 a_anchor）===
            # 约束方程：T = n·F_total - m·(n·a_anchor + v_t²/L + d²L/dt²)
            n_dot_a_anchor = np.dot(n, self.anchor_acc)
            term_centripetal = v_t_sq / max(self.rope_length, 1e-6)  # 向心加速度项
            term_rope_acc = self.rope_acc  # 绳长加速度项
            
            tension_candidate = (
                np.dot(n, total_force_world) 
                - self.mass * (n_dot_a_anchor + term_centripetal + term_rope_acc)
            )
            
            if tension_candidate <= 0:
                # 需要推力维持约束 → 绳松弛
                self.rope_taut = False
                self.tension = 0.0
                self.acc = total_force_world / self.mass
            else:
                # 绳绷紧，施加拉力
                self.tension = tension_candidate
                constraint_force = -self.tension * n  # 指向锚点
                self.acc = (total_force_world + constraint_force) / self.mass

    def _apply_impulse_to_match_rope_speed(self, v_rel, n, attach_offset_world, radial_vel):
        """冲量修正：使径向相对速度 = 绳速 (dL/dt)"""
        target_radial_vel = self.rope_speed
        delta_v = target_radial_vel - radial_vel
        if abs(delta_v) < 1e-6:
            return
        
        # 计算冲量传递系数
        inv_mass = 1.0 / self.mass
        gamma = self.inv_inertia @ np.cross(attach_offset_world, n)
        coupling = np.dot(n, np.cross(gamma, attach_offset_world))
        denom = inv_mass + coupling
        
        if abs(denom) < 1e-9:
            denom = inv_mass
            gamma = np.zeros(3)
        
        impulse = delta_v / denom
        self.vel += (impulse * inv_mass) * n
        self.omega += gamma * impulse

    def _quat_multiply(self, q1, q2):
        """四元数乘法 [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def publish_state(self):
        """发布仿真状态到 ROS2 topic"""
        # Odometry 消息（标准格式）
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id = "rigid_body"
        
        # 位置/姿态
        odom.pose.pose.position.x = self.pos[0]
        odom.pose.pose.position.y = self.pos[1]
        odom.pose.pose.position.z = self.pos[2]
        odom.pose.pose.orientation.x = self.quat[1]  # ROS2: [x,y,z,w]
        odom.pose.pose.orientation.y = self.quat[2]
        odom.pose.pose.orientation.z = self.quat[3]
        odom.pose.pose.orientation.w = self.quat[0]
        
        # 速度/角速度
        odom.twist.twist.linear.x = self.vel[0]
        odom.twist.twist.linear.y = self.vel[1]
        odom.twist.twist.linear.z = self.vel[2]
        odom.twist.twist.angular.x = self.omega[0]
        odom.twist.twist.angular.y = self.omega[1]
        odom.twist.twist.angular.z = self.omega[2]
        
        self.odom_pub.publish(odom)
        
        # 拉力
        tension_msg = Float64()
        tension_msg.data = self.tension
        self.tension_pub.publish(tension_msg)
        
        # 独立加速度消息（标准格式）
        accel_msg = AccelStamped()
        accel_msg.header.stamp = odom.header.stamp
        accel_msg.header.frame_id = "world"
        accel_msg.accel.linear.x = self.acc[0]
        accel_msg.accel.linear.y = self.acc[1]
        accel_msg.accel.linear.z = self.acc[2]
        accel_msg.accel.angular.x = self.alpha[0]
        accel_msg.accel.angular.y = self.alpha[1]
        accel_msg.accel.angular.z = self.alpha[2]
        self.accel_pub.publish(accel_msg)

def main(args=None):
    rclpy.init(args=args)
    simulator = AergripperSimulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()