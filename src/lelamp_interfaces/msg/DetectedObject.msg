# One object found in a camera frame, placed in base_link.

string name                        # remembered name, e.g. "potted plant"
float32 confidence                 # 0..1 from the detector
geometry_msgs/Point position       # estimated point in base_link (metres)
geometry_msgs/Vector3 direction    # unit ray from the camera to it, in base_link
float32[4] box                     # x1, y1, x2, y2 in image pixels
