import numpy as np
from scipy.spatial.transform import Rotation as R
from aerogripper_simulator.aerogripper_simulator.utils import LowPassFilter
from aerogripper_control.src.messages import ControllerState, PoseReference, Vector3, Quaternion

class AerogripperSimulator:
    """
    Simplified Aerogripper Simulator - Python wrapper.
    
    This provides a simplified physics simulation that works with the 
    tether observer model for position estimation.
    """
    
    def __init__(self, params):
        # === Parameter loading ===
        self.dt = params.get('dt', 0.01)
        self.publish_rate = params.get('publish_rate', 100)
        self.accel_filter_tau = params.get('accel_filter_tau', 0.1)
        self.mass = params.get('mass', 0.105)
        self.anchor_default_position = np.array(params.get('anchor_default_position', [0.0, 0.0, 0.4]))
        
        # === State initialization ===
        self._reset_state()
        
        # === Acceleration filters ===
        self.anchor_acc_filter = [
            LowPassFilter(self.accel_filter_tau, self.dt),
            LowPassFilter(self.accel_filter_tau, self.dt),
            LowPassFilter(self.accel_filter_tau, self.dt)
        ]
        self.rope_acc_filter = LowPassFilter(self.accel_filter_tau, self.dt)
        
        # === Input interfaces ===
        self.wrench_body = None
        self.thrust_body = np.zeros(3)
        self.tau_body = np.zeros(3)
        self.anchor_odom = None
        self.anchor_pos = self.anchor_default_position.copy()
        self.anchor_vel = np.zeros(3)
        self.rope_speed = 0.0
        
        # === Output interfaces ===
        self.simulator_odom = None
        self.simulator_tension = None
        self.simulator_accel = None
        
        # === Flight path history ===
        self.position_history = []  # List of (x, y, z) tuples
        
        # === Physics state ===
        self.position = self.anchor_default_position.copy()
        self.velocity = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        
    def _reset_state(self):
        """Reset simulation state."""
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.position_history = []
        
    def set_wrench_body(self, wrench):
        """Set body wrench (force and torque)."""
        self.wrench_body = wrench
        
    def set_thrust_and_torque(self, thrust, torque):
        """Set thrust and torque in body frame."""
        self.thrust_body = np.array(thrust) if thrust is not None else np.zeros(3)
        self.tau_body = np.array(torque) if torque is not None else np.zeros(3)
        
    def set_anchor_odom(self, odom):
        """Set anchor odometry."""
        self.anchor_odom = odom
        
    def set_anchor_state(self, pos, vel, acc=None):
        """Set anchor position and velocity."""
        self.anchor_pos = np.array(pos) if pos is not None else self.anchor_default_position.copy()
        self.anchor_vel = np.array(vel) if vel is not None else np.zeros(3)
        
    def set_rope_speed(self, speed):
        """Set rope speed."""
        self.rope_speed = speed if speed is not None else 0.0
        
    def simulation_step(self):
        """
        Execute one simulation step.
        
        Uses a simplified physics model that tracks the drone's position
        based on thrust and anchor movement.
        """
        # Calculate time
        t = len(self.position_history) * self.dt
        
        # Simplified physics: drone follows anchor with some lag
        # This is a placeholder that gets position from tether observer instead
        # For visualization purposes, we just record the position
        if len(self.position_history) == 0:
            self.position = self.anchor_pos.copy()
        else:
            # Simple kinematic model
            self.velocity += self.thrust_body * self.dt / self.mass
            self.velocity *= 0.95  # Damping
            
            # Add some drift toward anchor
            to_anchor = self.anchor_pos - self.position
            self.velocity += to_anchor * 0.1 * self.dt
            
            self.position += self.velocity * self.dt
            
        self.position_history.append(self.position.copy())
        
    def publish_state(self):
        """Prepare output data for visualization/logging."""
        pass
        
    def get_position(self):
        """Get current position."""
        return self.position
        
    def get_velocity(self):
        """Get current velocity."""
        return self.velocity

