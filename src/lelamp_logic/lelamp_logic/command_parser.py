"""Turn a spoken phrase into lamp commands, with a small local LLM on the CPU.

Plain Python (no ROS), so llm_node and the benchmark share it:

    parser = CommandParser('~/.lelamp/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf')
    parser.parse('lamp, look at the clock and turn blue', known=['clock', 'cup'])
    # -> [('look_at_object', 'clock'), ('set_light', 'blue')], 'llm'

Simple phrases ("look at the cup", "red light") are matched by keywords first
and never reach the model. Everything else goes to the LLM, whose output is
forced by a grammar into JSON that can only name these tools:

    look_at_object(object)   an object the lamp has seen
    scan_room()              look around the room
    look(direction)          left, right, up, down, forward
    move_to(posture)         rest, lean_in, sit_back, tall, shy
    set_light(color)         one of COLORS

A phrase that isn't a command gives an empty list.
"""
import json
import os
import re
import time

POSTURES = ['rest', 'lean_in', 'sit_back', 'tall', 'shy']
DIRECTIONS = ['left', 'right', 'up', 'down', 'forward']
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white',
          'warm', 'off']

SYSTEM_PROMPT = """You control a robot desk lamp. Turn what the user says into lamp commands.
Tools:
- look_at_object(object): look at an object the lamp has seen. Use the name from the seen list.
- scan_room(): look around the room.
- look(direction): left, right, up, down or forward.
- move_to(posture): rest, lean_in (come closer), sit_back (back off), tall (stand up), shy (hide, cower).
- set_light(color): red, orange, yellow, green, blue, purple, pink, white, warm, or off.
First decide if the user is asking the lamp to do something. Small talk, thanks,
questions about other things and "don't ..." are not requests.
Reply with JSON. Not a request: {"request": false}
A request: {"request": true, "commands": [...]} with only the commands asked for, in order.
Examples:
"look at my cup" -> {"request": true, "commands": [{"tool": "look_at_object", "object": "cup"}]}
"what's around you" -> {"request": true, "commands": [{"tool": "scan_room"}]}
"come here and go green" -> {"request": true, "commands": [{"tool": "move_to", "posture": "lean_in"}, {"tool": "set_light", "color": "green"}]}
"I'm getting coffee" -> {"request": false}
"thanks lamp" -> {"request": false}"""

ARG = {'look_at_object': 'object', 'look': 'direction', 'move_to': 'posture',
       'set_light': 'color', 'scan_room': None}


def _tool_schema(tool, arg=None, values=None):
    props = {'tool': {'type': 'string', 'enum': [tool]}}
    if arg:
        props[arg] = {'type': 'string', 'enum': values} if values else {'type': 'string'}
    return {'type': 'object', 'properties': props, 'required': list(props),
            'additionalProperties': False}


def make_schema(known=()):
    """{"request": false} or {"request": true, "commands": [1 to 3 commands]}.
    Deciding "request" first cuts down on small models inventing commands for
    small talk. look_at_object can only name objects in known (and is left out
    when nothing has been seen)."""
    tools = [_tool_schema('scan_room'),
             _tool_schema('look', 'direction', DIRECTIONS),
             _tool_schema('move_to', 'posture', POSTURES),
             _tool_schema('set_light', 'color', COLORS)]
    if known:
        tools.insert(0, _tool_schema('look_at_object', 'object', sorted(known)))
    return {'anyOf': [
        {'type': 'object', 'properties': {'request': {'type': 'boolean', 'enum': [False]}},
         'required': ['request'], 'additionalProperties': False},
        {'type': 'object',
         'properties': {'request': {'type': 'boolean', 'enum': [True]},
                        'commands': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                                     'items': {'anyOf': tools}}},
         'required': ['request', 'commands'], 'additionalProperties': False},
    ]}

def normalize(text):
    """Lower case, no punctuation, and whisper's 'lamb' fixed to 'lamp'."""
    t = text.lower().replace('-', ' ')
    t = re.sub(r"[^a-z0-9' ]+", ' ', t)
    t = re.sub(r'\blamb\b', 'lamp', t)
    return re.sub(r'\s+', ' ', t).strip()


# --- the LLM ---

class CommandParser:
    def __init__(self, model_path, n_threads=2, n_ctx=1024):
        from llama_cpp import Llama  # imported here so quick_parse works without it
        self.llm = Llama(model_path=os.path.expanduser(model_path), n_ctx=n_ctx,
                         n_threads=n_threads, n_threads_batch=n_threads,
                         n_gpu_layers=0, verbose=False)
        self.last_raw = ''
        self.parse('hello', use_keywords=False)  # warm up and cache the system prompt

    def ask_llm(self, text, known=()):
        seen = ', '.join(known) if known else 'nothing yet'
        out = self.llm.create_chat_completion(
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user', 'content': f'Seen objects: {seen}\nUser said: "{text}"'}],
            response_format={'type': 'json_object', 'schema': make_schema(known)},
            temperature=0.0, max_tokens=96)
        self.last_raw = out['choices'][0]['message']['content']
        try:
            items = json.loads(self.last_raw).get('commands', [])
        except (ValueError, AttributeError):
            return []
        cmds = []
        for item in items:
            tool = item.get('tool')
            cmd = (tool, str(item.get(ARG[tool], '')) if ARG.get(tool) else '')
            if tool in ARG and cmd not in cmds:  # small models sometimes repeat
                cmds.append(cmd)
        return cmds

    def parse(self, text, known=(), use_keywords=True):
        """(commands, source, seconds). commands is a list of (tool, arg);
        source is 'keywords' or 'llm'."""
        t0 = time.monotonic()
        cmds, source = self.ask_llm(normalize(text), known), 'llm'
        return cmds, source, time.monotonic() - t0
