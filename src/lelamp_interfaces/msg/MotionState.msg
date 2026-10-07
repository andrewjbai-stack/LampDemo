# What motion_node is doing, published every tick on /lelamp/motion_state.

uint32 goal_id               # last Look or Move goal accepted (0 = none yet)
bool settled                 # every joint has reached its target and stopped
float64 look_yaw             # where the head is aimed now, degrees (+ = lamp's left)
float64 look_pitch           # degrees (+ = up, 0 = level)
string look_mode             # "point" or "angle" (the last Look)
geometry_msgs/Point look_point  # the point, when look_mode is "point"
float64 shoulder             # posture target, radians
float64 elbow
