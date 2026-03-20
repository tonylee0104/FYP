# generated from rosidl_generator_py/resource/_idl.py.em
# with input from aerogripper_msgs:msg/ControllerState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'throttle_output'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ControllerState(type):
    """Metaclass of message 'ControllerState'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('aerogripper_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'aerogripper_msgs.msg.ControllerState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__controller_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__controller_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__controller_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__controller_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__controller_state

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

            from geometry_msgs.msg import Quaternion
            if Quaternion.__class__._TYPE_SUPPORT is None:
                Quaternion.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ControllerState(metaclass=Metaclass_ControllerState):
    """Message class 'ControllerState'."""

    __slots__ = [
        '_timestamp',
        '_dt',
        '_desired_acceleration',
        '_desired_angular_velocity',
        '_desired_thrust',
        '_desired_torque',
        '_current_position',
        '_current_velocity',
        '_current_orientation',
        '_current_euler_angles',
        '_current_angular_velocity',
        '_reference_position',
        '_reference_velocity',
        '_reference_orientation',
        '_reference_euler_angles',
        '_throttle_output',
        '_tether_force_bodyframe',
    ]

    _fields_and_field_types = {
        'timestamp': 'builtin_interfaces/Time',
        'dt': 'double',
        'desired_acceleration': 'geometry_msgs/Vector3',
        'desired_angular_velocity': 'geometry_msgs/Vector3',
        'desired_thrust': 'geometry_msgs/Vector3',
        'desired_torque': 'geometry_msgs/Vector3',
        'current_position': 'geometry_msgs/Point',
        'current_velocity': 'geometry_msgs/Vector3',
        'current_orientation': 'geometry_msgs/Quaternion',
        'current_euler_angles': 'geometry_msgs/Vector3',
        'current_angular_velocity': 'geometry_msgs/Vector3',
        'reference_position': 'geometry_msgs/Point',
        'reference_velocity': 'geometry_msgs/Vector3',
        'reference_orientation': 'geometry_msgs/Quaternion',
        'reference_euler_angles': 'geometry_msgs/Vector3',
        'throttle_output': 'double[4]',
        'tether_force_bodyframe': 'geometry_msgs/Vector3',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Quaternion'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Quaternion'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from builtin_interfaces.msg import Time
        self.timestamp = kwargs.get('timestamp', Time())
        self.dt = kwargs.get('dt', float())
        from geometry_msgs.msg import Vector3
        self.desired_acceleration = kwargs.get('desired_acceleration', Vector3())
        from geometry_msgs.msg import Vector3
        self.desired_angular_velocity = kwargs.get('desired_angular_velocity', Vector3())
        from geometry_msgs.msg import Vector3
        self.desired_thrust = kwargs.get('desired_thrust', Vector3())
        from geometry_msgs.msg import Vector3
        self.desired_torque = kwargs.get('desired_torque', Vector3())
        from geometry_msgs.msg import Point
        self.current_position = kwargs.get('current_position', Point())
        from geometry_msgs.msg import Vector3
        self.current_velocity = kwargs.get('current_velocity', Vector3())
        from geometry_msgs.msg import Quaternion
        self.current_orientation = kwargs.get('current_orientation', Quaternion())
        from geometry_msgs.msg import Vector3
        self.current_euler_angles = kwargs.get('current_euler_angles', Vector3())
        from geometry_msgs.msg import Vector3
        self.current_angular_velocity = kwargs.get('current_angular_velocity', Vector3())
        from geometry_msgs.msg import Point
        self.reference_position = kwargs.get('reference_position', Point())
        from geometry_msgs.msg import Vector3
        self.reference_velocity = kwargs.get('reference_velocity', Vector3())
        from geometry_msgs.msg import Quaternion
        self.reference_orientation = kwargs.get('reference_orientation', Quaternion())
        from geometry_msgs.msg import Vector3
        self.reference_euler_angles = kwargs.get('reference_euler_angles', Vector3())
        if 'throttle_output' not in kwargs:
            self.throttle_output = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.throttle_output = kwargs.get('throttle_output')
        from geometry_msgs.msg import Vector3
        self.tether_force_bodyframe = kwargs.get('tether_force_bodyframe', Vector3())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.timestamp != other.timestamp:
            return False
        if self.dt != other.dt:
            return False
        if self.desired_acceleration != other.desired_acceleration:
            return False
        if self.desired_angular_velocity != other.desired_angular_velocity:
            return False
        if self.desired_thrust != other.desired_thrust:
            return False
        if self.desired_torque != other.desired_torque:
            return False
        if self.current_position != other.current_position:
            return False
        if self.current_velocity != other.current_velocity:
            return False
        if self.current_orientation != other.current_orientation:
            return False
        if self.current_euler_angles != other.current_euler_angles:
            return False
        if self.current_angular_velocity != other.current_angular_velocity:
            return False
        if self.reference_position != other.reference_position:
            return False
        if self.reference_velocity != other.reference_velocity:
            return False
        if self.reference_orientation != other.reference_orientation:
            return False
        if self.reference_euler_angles != other.reference_euler_angles:
            return False
        if any(self.throttle_output != other.throttle_output):
            return False
        if self.tether_force_bodyframe != other.tether_force_bodyframe:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def timestamp(self):
        """Message field 'timestamp'."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'timestamp' field must be a sub message of type 'Time'"
        self._timestamp = value

    @builtins.property
    def dt(self):
        """Message field 'dt'."""
        return self._dt

    @dt.setter
    def dt(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'dt' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'dt' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._dt = value

    @builtins.property
    def desired_acceleration(self):
        """Message field 'desired_acceleration'."""
        return self._desired_acceleration

    @desired_acceleration.setter
    def desired_acceleration(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'desired_acceleration' field must be a sub message of type 'Vector3'"
        self._desired_acceleration = value

    @builtins.property
    def desired_angular_velocity(self):
        """Message field 'desired_angular_velocity'."""
        return self._desired_angular_velocity

    @desired_angular_velocity.setter
    def desired_angular_velocity(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'desired_angular_velocity' field must be a sub message of type 'Vector3'"
        self._desired_angular_velocity = value

    @builtins.property
    def desired_thrust(self):
        """Message field 'desired_thrust'."""
        return self._desired_thrust

    @desired_thrust.setter
    def desired_thrust(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'desired_thrust' field must be a sub message of type 'Vector3'"
        self._desired_thrust = value

    @builtins.property
    def desired_torque(self):
        """Message field 'desired_torque'."""
        return self._desired_torque

    @desired_torque.setter
    def desired_torque(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'desired_torque' field must be a sub message of type 'Vector3'"
        self._desired_torque = value

    @builtins.property
    def current_position(self):
        """Message field 'current_position'."""
        return self._current_position

    @current_position.setter
    def current_position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'current_position' field must be a sub message of type 'Point'"
        self._current_position = value

    @builtins.property
    def current_velocity(self):
        """Message field 'current_velocity'."""
        return self._current_velocity

    @current_velocity.setter
    def current_velocity(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'current_velocity' field must be a sub message of type 'Vector3'"
        self._current_velocity = value

    @builtins.property
    def current_orientation(self):
        """Message field 'current_orientation'."""
        return self._current_orientation

    @current_orientation.setter
    def current_orientation(self, value):
        if __debug__:
            from geometry_msgs.msg import Quaternion
            assert \
                isinstance(value, Quaternion), \
                "The 'current_orientation' field must be a sub message of type 'Quaternion'"
        self._current_orientation = value

    @builtins.property
    def current_euler_angles(self):
        """Message field 'current_euler_angles'."""
        return self._current_euler_angles

    @current_euler_angles.setter
    def current_euler_angles(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'current_euler_angles' field must be a sub message of type 'Vector3'"
        self._current_euler_angles = value

    @builtins.property
    def current_angular_velocity(self):
        """Message field 'current_angular_velocity'."""
        return self._current_angular_velocity

    @current_angular_velocity.setter
    def current_angular_velocity(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'current_angular_velocity' field must be a sub message of type 'Vector3'"
        self._current_angular_velocity = value

    @builtins.property
    def reference_position(self):
        """Message field 'reference_position'."""
        return self._reference_position

    @reference_position.setter
    def reference_position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'reference_position' field must be a sub message of type 'Point'"
        self._reference_position = value

    @builtins.property
    def reference_velocity(self):
        """Message field 'reference_velocity'."""
        return self._reference_velocity

    @reference_velocity.setter
    def reference_velocity(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'reference_velocity' field must be a sub message of type 'Vector3'"
        self._reference_velocity = value

    @builtins.property
    def reference_orientation(self):
        """Message field 'reference_orientation'."""
        return self._reference_orientation

    @reference_orientation.setter
    def reference_orientation(self, value):
        if __debug__:
            from geometry_msgs.msg import Quaternion
            assert \
                isinstance(value, Quaternion), \
                "The 'reference_orientation' field must be a sub message of type 'Quaternion'"
        self._reference_orientation = value

    @builtins.property
    def reference_euler_angles(self):
        """Message field 'reference_euler_angles'."""
        return self._reference_euler_angles

    @reference_euler_angles.setter
    def reference_euler_angles(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'reference_euler_angles' field must be a sub message of type 'Vector3'"
        self._reference_euler_angles = value

    @builtins.property
    def throttle_output(self):
        """Message field 'throttle_output'."""
        return self._throttle_output

    @throttle_output.setter
    def throttle_output(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'throttle_output' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'throttle_output' numpy.ndarray() must have a size of 4"
            self._throttle_output = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'throttle_output' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._throttle_output = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def tether_force_bodyframe(self):
        """Message field 'tether_force_bodyframe'."""
        return self._tether_force_bodyframe

    @tether_force_bodyframe.setter
    def tether_force_bodyframe(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'tether_force_bodyframe' field must be a sub message of type 'Vector3'"
        self._tether_force_bodyframe = value
