# One face found in a camera frame.

float32[4] box                     # x, y, width, height in image pixels
float32 score                      # 0..1 from YuNet
float32[10] landmarks              # (x, y) pixels: right eye, left eye, nose tip,
                                   # right mouth corner, left mouth corner (the person's)
float32 yaw                        # degrees, + = head turned toward image right
float32 pitch                      # degrees, + = head tilted up
float32 off_axis                   # degrees between head direction and the line to the camera
geometry_msgs/Point center         # x, y in [-1, 1] (+x right, +y down); z = width / image width
