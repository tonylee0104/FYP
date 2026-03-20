make hkust_nxt-dual

MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600

ros2 launch aerogripper_control test.launch.py