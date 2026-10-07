"""Turns heard phrases into lamp commands: a small LLM on the CPU, behind a service.

logic_node sends each phrase voice_node hears, plus the names of the objects it
remembers, and gets back the commands to run. Every phrase goes to Qwen2.5
through llama.cpp (command_parser.py).

    ros2 service call /lelamp/parse_command lelamp_interfaces/srv/ParseCommand \\
        "{text: 'lamp, look at the clock and turn blue', known_objects: [clock]}"

Service: /lelamp/parse_command (lelamp_interfaces/ParseCommand)
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from lelamp_interfaces.msg import LampCommand
from lelamp_interfaces.srv import ParseCommand
from lelamp_logic.command_parser import CommandParser


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self.declare_parameter('model_path',
                               '~/.lelamp/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf')
        self.declare_parameter('threads', 2)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.parser = None
        try:
            self.parser = CommandParser(gp('model_path'), n_threads=int(gp('threads')))
        except Exception as e:  # missing llama_cpp or model
            self.get_logger().error(f'LLM failed to load, every call will fail: {e}')
        self.create_service(ParseCommand, 'lelamp/parse_command', self._on_parse)
        if self.parser:
            self.get_logger().info('LLM node ready')

    def _on_parse(self, req, res):
        if self.parser is None:
            res.success, res.message = False, 'LLM not loaded'
            return res
        cmds, res.source, res.seconds = self.parser.parse(req.text, list(req.known_objects))
        res.commands = [LampCommand(tool=t, arg=a) for t, a in cmds]
        res.success = True
        res.message = self.parser.last_raw
        self.get_logger().info(f'"{req.text}" -> {cmds or "nothing"} '
                               f'({res.source}, {res.seconds:.2f}s)')
        return res


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
