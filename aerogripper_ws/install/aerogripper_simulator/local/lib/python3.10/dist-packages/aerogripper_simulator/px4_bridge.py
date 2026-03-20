#!/usr/bin/env python3
"""
Bridge between aerogripper simulator and PX4-style topics.
- ActuatorMotors -> Wrench (body)
- Odometry/Accel -> VehicleAttitude/SensorCombined
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Wrench, AccelStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import ActuatorMotors, VehicleAttitude, SensorCombined


class Px4Bridge(Node):
    def __init__(self):
        super().__init__("aerogripper_px4_bridge")

        # Parameters (keep defaults aligned with controller config where possible)
        self.declare_parameter("thrust_coeff", 6.503575)
        self.declare_parameter("thrust_arm_length", 0.055)
        self.declare_parameter("torque_coeff", 0.002)

        self.thrust_coeff = float(self.get_parameter("thrust_coeff").value)
        self.thrust_arm_length = float(self.get_parameter("thrust_arm_length").value)
        self.torque_coeff = float(self.get_parameter("torque_coeff").value)

        self._build_allocation_matrix()

        # Subscribers
        self.create_subscription(ActuatorMotors, "/fmu/in/actuator_motors", self.actuator_cb, 10)
        self.create_subscription(Odometry, "simulator/odom", self.odom_cb, 10)
        self.create_subscription(AccelStamped, "simulator/accel", self.accel_cb, 10)

        px4_qos = QoSProfile(depth=1)
        px4_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        px4_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # Publishers
        self.wrench_pub = self.create_publisher(Wrench, "wrench_body", 10)
        self.attitude_pub = self.create_publisher(VehicleAttitude, "/fmu/out/vehicle_attitude", px4_qos)
        self.imu_pub = self.create_publisher(SensorCombined, "/fmu/out/sensor_combined", px4_qos)

        # State cache
        self.last_q_flu = None
        self.last_omega_flu = None

        self.get_logger().info("PX4 bridge initialized")

    def _build_allocation_matrix(self):
        l_t = self.thrust_arm_length
        kappa_t = self.torque_coeff
        sqrt2_2 = math.sqrt(2.0) / 2.0

        # Same as Mixer::allocation_matrix_ (6x4)
        self.allocation_matrix = np.array([
            [-0.5,           0.5,          -0.5,           0.5],
            [ 0.5,          -0.5,          -0.5,           0.5],
            [-sqrt2_2,      -sqrt2_2,      -sqrt2_2,      -sqrt2_2],
            [ 0.5*l_t - 0.5*kappa_t,  -0.5*l_t + 0.5*kappa_t,  -0.5*l_t + 0.5*kappa_t,   0.5*l_t - 0.5*kappa_t],
            [ 0.5*l_t + 0.5*kappa_t,  -0.5*l_t - 0.5*kappa_t,   0.5*l_t + 0.5*kappa_t,  -0.5*l_t - 0.5*kappa_t],
            [-sqrt2_2*kappa_t,  -sqrt2_2*kappa_t,   sqrt2_2*kappa_t,   sqrt2_2*kappa_t]
        ], dtype=np.float64)

    def actuator_cb(self, msg: ActuatorMotors):
        # Extract first 4 motors, guard against NaN
        motors = np.array(msg.control[:4], dtype=np.float64)
        motors = np.where(np.isfinite(motors), motors, 0.0)
        motors = np.clip(motors, 0.0, None)

        thrust = self.thrust_coeff * motors * motors
        control = self.allocation_matrix @ thrust

        # allocation matrix output is FRD body frame, simulator expects FLU.
        force_frd = np.array([control[0], control[1], control[2]], dtype=np.float64)
        torque_frd = np.array([control[3], control[4], control[5]], dtype=np.float64)
        force_flu = np.array([force_frd[0], -force_frd[1], -force_frd[2]], dtype=np.float64)
        torque_flu = np.array([torque_frd[0], -torque_frd[1], -torque_frd[2]], dtype=np.float64)

        wrench = Wrench()
        wrench.force.x = float(force_flu[0])
        wrench.force.y = float(force_flu[1])
        wrench.force.z = float(force_flu[2])
        wrench.torque.x = float(torque_flu[0])
        wrench.torque.y = float(torque_flu[1])
        wrench.torque.z = float(torque_flu[2])
        self.wrench_pub.publish(wrench)

    def odom_cb(self, msg: Odometry):
        # FLU quaternion from simulator
        q_flu = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ], dtype=np.float64)

        # FLU -> FRD conversion: q_frd = q_x180 * q_flu * q_x180
        q_x180 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        q_frd = self._quat_multiply(self._quat_multiply(q_x180, q_flu), q_x180)

        att = VehicleAttitude()
        att.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        att.q[0] = float(q_frd[0])
        att.q[1] = float(q_frd[1])
        att.q[2] = float(q_frd[2])
        att.q[3] = float(q_frd[3])
        self.attitude_pub.publish(att)

        # Cache for IMU conversion
        self.last_q_flu = q_flu
        self.last_omega_flu = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=np.float64)

    def accel_cb(self, msg: AccelStamped):
        if self.last_q_flu is None or self.last_omega_flu is None:
            return

        acc_world = np.array([
            msg.accel.linear.x,
            msg.accel.linear.y,
            msg.accel.linear.z,
        ], dtype=np.float64)

        # Build rotation matrix from q_flu (body->world)
        R_bw = self._quat_to_rot(self.last_q_flu).T  # world->body
        g_world = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

        acc_body = R_bw @ acc_world
        g_body = R_bw @ g_world
        specific_force_body = acc_body - g_body

        # FLU -> FRD
        acc_frd = np.array([specific_force_body[0], -specific_force_body[1], -specific_force_body[2]])
        omega_frd = np.array([self.last_omega_flu[0], -self.last_omega_flu[1], -self.last_omega_flu[2]])

        imu = SensorCombined()
        imu.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        imu.accelerometer_timestamp_relative = 0
        imu.accelerometer_integral_dt = 0
        imu.gyro_integral_dt = 0
        imu.accelerometer_clipping = 0
        imu.gyro_clipping = 0
        imu.accel_calibration_count = 0
        imu.gyro_calibration_count = 0
        imu.accelerometer_m_s2[0] = float(acc_frd[0])
        imu.accelerometer_m_s2[1] = float(acc_frd[1])
        imu.accelerometer_m_s2[2] = float(acc_frd[2])
        imu.gyro_rad[0] = float(omega_frd[0])
        imu.gyro_rad[1] = float(omega_frd[1])
        imu.gyro_rad[2] = float(omega_frd[2])
        self.imu_pub.publish(imu)

    def _quat_to_rot(self, q):
        # q = [w, x, y, z]
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _quat_multiply(self, q1, q2):
        # [w, x, y, z]
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = Px4Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
