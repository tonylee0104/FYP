"""
Unit tests for the Mixer class.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'aerogripper_control', 'src'))

from mixer import Mixer


class TestMixer:
    """Test cases for the Mixer class."""
    
    def test_mixer_initialization_default(self):
        """Test mixer initialization with default parameters."""
        mixer = Mixer()
        assert mixer.theta == np.deg2rad(45)
        assert mixer.thrust_arm_length == 0.2
        assert mixer.gravity_arm_length == 0.2
        
    def test_mixer_initialization_custom_angle(self):
        """Test mixer initialization with custom motor angle."""
        mixer = Mixer(theta_deg=60)
        assert mixer.theta == np.deg2rad(60)
        
    def test_mixer_initialization_invalid_angle_negative(self):
        """Test mixer initialization with invalid negative angle."""
        with pytest.raises(ValueError, match="theta_deg must be between 0 and 90"):
            Mixer(theta_deg=-10)
            
    def test_mixer_initialization_invalid_angle_too_large(self):
        """Test mixer initialization with angle > 90."""
        with pytest.raises(ValueError, match="theta_deg must be between 0 and 90"):
            Mixer(theta_deg=100)
            
    def test_mixer_initialization_angle_0(self):
        """Test mixer initialization with 0 degree angle."""
        mixer = Mixer(theta_deg=0)
        assert mixer.theta == 0
        
    def test_mixer_initialization_angle_90(self):
        """Test mixer initialization with 90 degree angle."""
        mixer = Mixer(theta_deg=90)
        assert mixer.theta == np.deg2rad(90)
        
    def test_allocation_matrix_shape(self):
        """Test that allocation matrix has correct shape."""
        mixer = Mixer()
        assert mixer.allocation_matrix.shape == (6, 4)
        
    def test_mix_matrix_shape(self):
        """Test that mix matrix has correct shape."""
        mixer = Mixer()
        assert mixer.mix_matrix.shape == (4, 6)
        
    def test_control_allocation_shape(self):
        """Test control_allocation input/output shape."""
        mixer = Mixer()
        motors_thrust = np.array([1.0, 1.0, 1.0, 1.0])
        result = mixer.control_allocation(motors_thrust)
        assert result.shape == (6,)
        
    def test_mix_shape(self):
        """Test mix input/output shape."""
        mixer = Mixer()
        u = np.array([0.1, 0.1, 1.0, 0.0, 0.0, 0.0])  # desired wrench
        result = mixer.mix(u)
        assert result.shape == (4,)
        
    def test_throttle2thrust_shape(self):
        """Test throttle2thrust input/output shape."""
        mixer = Mixer()
        throttle = np.array([0.5, 0.5, 0.5, 0.5])
        result = mixer.throttle2thrust(throttle)
        assert result.shape == (4,)
        
    def test_throttle2thrust_values(self):
        """Test throttle2thrust computation."""
        mixer = Mixer(thrust_coeff=0.01)
        throttle = np.array([1.0, 1.0, 1.0, 1.0])
        result = mixer.throttle2thrust(throttle)
        # thrust = coeff * throttle^2 = 0.01 * 1.0 = 0.01
        expected = np.array([0.01, 0.01, 0.01, 0.01])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_thrust2throttle_shape(self):
        """Test thrust2throttle input/output shape."""
        mixer = Mixer()
        thrust = np.array([0.01, 0.01, 0.01, 0.01])
        result = mixer.thrust2throttle(thrust)
        assert result.shape == (4,)
        
    def test_thrust2throttle_values(self):
        """Test thrust2throttle computation."""
        mixer = Mixer(thrust_coeff=0.01)
        thrust = np.array([0.01, 0.01, 0.01, 0.01])
        result = mixer.thrust2throttle(thrust)
        # throttle = sqrt(thrust / coeff) = sqrt(0.01 / 0.01) = 1.0
        expected = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_set_params_valid(self):
        """Test set_params with valid parameters."""
        mixer = Mixer()
        mixer.set_params(
            thrust_arm_length=0.1,
            gravity_arm_length=0.1,
            torque_coeff=0.02,
            rpm_coeff=0.02,
            thrust_coeff=0.02,
            yaw_weight=2.0,
            attitude_weight=2.0,
            thrust_weight=2.0
        )
        assert mixer.thrust_arm_length == 0.1
        assert mixer.gravity_arm_length == 0.1
        assert mixer.thrust_coeff == 0.02
        
    def test_set_params_invalid_thrust_arm_length(self):
        """Test set_params with invalid thrust_arm_length."""
        mixer = Mixer()
        with pytest.raises(ValueError, match="thrust_arm_length must be positive"):
            mixer.set_params(
                thrust_arm_length=-0.1,
                gravity_arm_length=0.1,
                torque_coeff=0.01,
                rpm_coeff=0.01,
                thrust_coeff=0.01,
                yaw_weight=1.0,
                attitude_weight=1.0,
                thrust_weight=1.0
            )
            
    def test_set_params_invalid_thrust_coeff(self):
        """Test set_params with invalid thrust_coeff."""
        mixer = Mixer()
        with pytest.raises(ValueError, match="thrust_coeff must be positive"):
            mixer.set_params(
                thrust_arm_length=0.1,
                gravity_arm_length=0.1,
                torque_coeff=0.01,
                rpm_coeff=0.01,
                thrust_coeff=-0.01,
                yaw_weight=1.0,
                attitude_weight=1.0,
                thrust_weight=1.0
            )
            
    def test_set_params_with_theta(self):
        """Test set_params with theta_deg parameter."""
        mixer = Mixer()
        mixer.set_params(
            thrust_arm_length=0.1,
            gravity_arm_length=0.1,
            torque_coeff=0.01,
            rpm_coeff=0.01,
            thrust_coeff=0.01,
            yaw_weight=1.0,
            attitude_weight=1.0,
            thrust_weight=1.0,
            theta_deg=60
        )
        assert mixer.theta == np.deg2rad(60)
        
    def test_motor_angle_effects_allocation(self):
        """Test that different motor angles produce different allocation matrices."""
        mixer_30 = Mixer(theta_deg=30)
        mixer_60 = Mixer(theta_deg=60)
        
        # Matrices should be different
        assert not np.allclose(mixer_30.allocation_matrix, mixer_60.allocation_matrix)
        
    def test_zero_angle_vertical_thrust(self):
        """Test that 0 degree angle gives pure vertical thrust."""
        mixer = Mixer(theta_deg=0)
        
        # At 0 degrees, all thrust should be in Z direction
        motors_thrust = np.array([1.0, 1.0, 1.0, 1.0])
        wrench = mixer.control_allocation(motors_thrust)
        
        # X and Y forces should be zero at 0 degrees
        assert abs(wrench[0]) < 1e-10  # Fx
        assert abs(wrench[1]) < 1e-10  # Fy
        
    def test_roundtrip_throttle_thrust(self):
        """Test that throttle->thrust->throttle roundtrip is consistent."""
        mixer = Mixer(thrust_coeff=0.01)
        
        original_throttle = np.array([0.5, 0.7, 0.3, 0.9])
        thrust = mixer.throttle2thrust(original_throttle)
        recovered_throttle = mixer.thrust2throttle(thrust)
        
        np.testing.assert_array_almost_equal(original_throttle, recovered_throttle, decimal=10)


class TestMixerParameterSweep:
    """Test mixer behavior across different parameter ranges."""
    
    def test_angle_sweep(self):
        """Test mixer across a range of angles."""
        angles = [0, 15, 30, 45, 60, 75, 90]
        matrices = []
        
        for angle in angles:
            mixer = Mixer(theta_deg=angle)
            matrices.append(mixer.allocation_matrix.copy())
            
        # All matrices should be different
        for i in range(len(matrices)):
            for j in range(i+1, len(matrices)):
                assert not np.allclose(matrices[i], matrices[j]), f"Matrices at {angles[i]}° and {angles[j]}° should differ"
                
    def test_motor_configuration_consistency(self):
        """Test that the same angle produces consistent results."""
        mixer1 = Mixer(theta_deg=45)
        mixer2 = Mixer(theta_deg=45)
        
        test_input = np.array([0.5, 0.5, 0.5, 0.5])
        result1 = mixer1.control_allocation(test_input)
        result2 = mixer2.control_allocation(test_input)
        
        np.testing.assert_array_almost_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

