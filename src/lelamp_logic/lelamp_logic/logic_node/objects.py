"""Object checks for logic_node.

While looking around, the lamp asks object_node what it can see (the
detect_objects service) and remembers where each object is in base_link
(object_memory.py). Asking for an object by name makes it look back at it.

Input: /lelamp/look_at_object (std_msgs/String, e.g. "clock")
Uses:  /lelamp/detect_objects (lelamp_interfaces/DetectObjects, object_node)
"""
from std_msgs.msg import String

from lelamp_interfaces.srv import DetectObjects
from lelamp_logic.logic_node.object_memory import ObjectMemory


class ObjectsMixin:
    def _init_objects(self):
        self.memory = ObjectMemory()  # fresh every run
        self.focus_target = None  # (x, y, z) the focus behavior looks at
        self.detect_client = self.create_client(DetectObjects, 'lelamp/detect_objects')
        self.detecting = False  # a detect_objects call is in flight
        self.create_subscription(String, 'lelamp/look_at_object', self._on_look_at_object, 10)

    def _on_look_at_object(self, msg):
        self.look_at_object(msg.data)

    def snapshot(self):
        """Ask object_node what's in view now and remember it. Returns at once
        (the answer arrives in _on_detected); False if skipped."""
        if self.detecting or self.thinking or not self.detect_client.service_is_ready():
            return False
        self.detecting = True
        self.detect_client.call_async(DetectObjects.Request()).add_done_callback(self._on_detected)
        return True

    def _on_detected(self, future):
        self.detecting = False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f'detect_objects failed: {e}')
            return
        if not res.success:
            self.get_logger().warn(f'detect_objects: {res.message}', throttle_duration_sec=5.0)
            return
        for obj in res.objects:
            p = obj.position
            self.memory.add(obj.name, (p.x, p.y, p.z), obj.confidence)
        if res.objects:
            self.get_logger().info(f'saw: {res.message}')

    def look_at_object(self, name):
        """Look at a remembered object for a while. False if it's never been seen."""
        point = self.memory.find(name)
        if point is None:
            self.get_logger().info(f"haven't seen '{name}' (known: {', '.join(self.memory.names())})")
            return False
        self.focus_target = point
        self.brain.go('focus')
        return True
