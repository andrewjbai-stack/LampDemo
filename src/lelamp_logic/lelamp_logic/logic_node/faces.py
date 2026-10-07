"""Face checks for logic_node.

face_rate_hz times a second it asks face_node whether someone is looking at
the lamp (the detect_faces service); that sets attentive.

Uses: /lelamp/detect_faces (lelamp_interfaces/DetectFaces, face_node)
"""
from lelamp_interfaces.srv import DetectFaces


class FacesMixin:
    def _init_faces(self, rate_hz):
        self.attentive = False  # When user looks at lamp/webcam, lamp becomes attentive
        self.faces = []  # latest DetectedFace list from face_node, largest first
        self.face_client = self.create_client(DetectFaces, 'lelamp/detect_faces')
        self.checking_faces = False  # a detect_faces call is in flight
        self.create_timer(1.0 / rate_hz, self.check_faces)

    def check_faces(self):
        """Ask face_node about the newest frame; the answer sets attentive."""
        if self.checking_faces or self.thinking or not self.face_client.service_is_ready():
            return
        self.checking_faces = True
        self.face_client.call_async(DetectFaces.Request()).add_done_callback(self._on_faces)

    def _on_faces(self, future):
        self.checking_faces = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'detect_faces failed: {e}', throttle_duration_sec=5.0)
            return
        if not res.success:
            return
        self.faces = res.faces
        self.attentive = res.looking
