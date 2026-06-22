#!/usr/bin/env python3
import threading
from collections import deque

import cv2
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Empty
from sensor_msgs.msg import CompressedImage


class VLABaseNode(Node):
    """Base node for VLA navigation models: step-synchronous discrete-action loop."""

    def __init__(self, node_name, vla_model):
        super().__init__(node_name)

        # ---- parameters ----
        self.declare_parameter("image_topic", "")
        self.declare_parameter("goal_topic", "/goal_instruction")
        self.declare_parameter("action_topic", f"/{vla_model}/action")
        self.declare_parameter("reset_topic", f"/{vla_model}/reset")
        self.declare_parameter("status_topic", f"/{vla_model}/primitive_status")
        self._declare_params()

        prm = lambda n: self.get_parameter(n).value

        image_topic = prm("image_topic")
        goal_topic = prm("goal_topic")
        action_topic = prm("action_topic")
        reset_topic = prm("reset_topic")
        status_topic = prm("status_topic")

        # ---- state ----
        self._agent = None
        self._model_ready = False
        self._lock = threading.Lock()
        self._last_image_msg = None
        self._goal = None
        self._queue = deque()
        self._busy = False

        # ---- I/O ----
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_image = self.create_subscription(CompressedImage, image_topic, self._image_cb, qos_sensor)
        self.sub_goal = self.create_subscription(String, goal_topic, self._goal_cb, 10)
        self.sub_reset = self.create_subscription(Empty, reset_topic, self._reset_cb, 10)
        self.sub_status = self.create_subscription(String, status_topic, self._status_cb, 10)
        self.pub_action = self.create_publisher(String, action_topic, 10)

        threading.Thread(target=self._load_model_thread, daemon=True).start()

    # ================= To implement in subclass =================
    def _declare_params(self):
        """Declare model-specific parameters (e.g. model_path). Override if needed."""
        pass

    def load_model(self):
        """Build and return the VLA agent (download weights if needed)."""
        raise NotImplementedError

    def infer_action(self, frame, goal):
        """Return a list of discrete action tokens for the given frame/goal."""
        raise NotImplementedError

    def reset_agent(self):
        """Reset the agent's internal history. Override if needed."""
        if self._agent is not None:
            self._agent.reset()

    # ================= Model loading =================
    def _load_model_thread(self):
        try:
            agent = self.load_model()
            with self._lock:
                self._agent = agent
                self._model_ready = True
            self.get_logger().info(f"{self.get_name()} ready - waiting for goal instruction")
        except Exception as e:
            self.get_logger().error(f"Error while loading model: {e}")

    # ================= Callbacks =================
    def _image_cb(self, msg: CompressedImage):
        with self._lock:
            self._last_image_msg = msg

    def _goal_cb(self, msg: String):
        goal = msg.data.strip()
        if not goal:
            return
        self._goal = goal
        self._busy = False
        self.get_logger().info(f"New goal: {goal}")
        self._step()                      # no memory reset


    def _reset_cb(self, msg: Empty):
        self._goal = None
        self._reset()
        self.get_logger().info("Reset")

    def _status_cb(self, msg: String):
        if msg.data.strip().lower() in ("done", "aborted", "idle", "ready"):
            self._busy = False
            self._step()            

    # ================= Step-synchronous loop =================
    def _step(self):
        if self._busy or self._goal is None or not self._model_ready:
            return
        frame = self._decode_latest()
        if frame is None:
            return
        actions = self.infer_action(frame, self._goal)
        if not actions:
            return
        action = actions[0]               # execute-first: il chunk è un hint d'orizzonte
        self._busy = True
        self._publish_action(action)
        if action == "stop":
            self._on_stop()
    
    def _on_stop(self):
        """Called when the model emits 'stop'. Override for task-specific behavior."""
        self._goal = None
        self.get_logger().info("STOP reached")

    # ================= Helpers =================
    def _publish_action(self, cmd: str):
        out = String()
        out.data = cmd
        self.pub_action.publish(out)

    def _reset(self):
        with self._lock:
            self.reset_agent()
        self._queue.clear()
        self._busy = False

    def _decode_latest(self):
        with self._lock:
            msg = self._last_image_msg
        if msg is None:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)