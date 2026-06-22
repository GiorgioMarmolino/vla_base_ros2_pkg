#!/usr/bin/env python3
"""
action_node.py — executes discrete VLA primitives as closed-loop motions.

Each primitive has a FIXED magnitude (Uni-NaVid emits chunks of unit steps,
not a variable distance like NaVILA did):
    forward            -> drive forward  `forward_step_m`
    left  / turn_left  -> rotate +`turn_step_deg`
    right / turn_right -> rotate -`turn_step_deg`
    stop               -> halt, report done

The VLA node queues the 4-action chunk and sends one token per
`primitive_status` round-trip; this node executes a single primitive at a time
on odometry and publishes 'done'/'aborted'. On `path_blocked` a forward
primitive is aborted so the VLA can re-plan from the new frame.

Output topic is parametric so the launch file can route around the safety node:
    safety ON  -> /cmd_vel_raw   (safety node consumes it)
    safety OFF -> /cmd_vel       (straight to twist_mux)

Subscribes: <action_topic> (String), <odom_topic> (Odometry),
            <path_blocked_topic> (Bool)
Publishes:  <cmd_vel_topic> (Twist), <status_topic> (String)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_FORWARD_STEP_M = 0.25   # m      — fixed forward step
DEFAULT_TURN_STEP_DEG  = 25.0   # deg    — fixed turn step

DEFAULT_LINEAR_X       = 0.4    # m/s    — forward execution speed
DEFAULT_ANGULAR_Z      = 0.35   # rad/s  — turn execution speed

DEFAULT_PUBLISH_RATE   = 0.05   # s      — publish period (20 Hz)
DEFAULT_MAX_ACC_LIN    = 1.0    # m/s^2
DEFAULT_MAX_ACC_ANG    = 2.0    # rad/s^2

FORWARD_TOL_M   = 0.01          # m   — stop slightly early
TURN_TOL_RAD    = math.radians(1.0)


class ActionNode(Node):

    def __init__(self):
        super().__init__("action_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("action_topic",       "/uninavid/action")
        self.declare_parameter("cmd_vel_topic",      "/cmd_vel")
        self.declare_parameter("odom_topic",         "/platform/odom/filtered")
        self.declare_parameter("status_topic",       "/uninavid/primitive_status")
        self.declare_parameter("path_blocked_topic", "/safety/path_blocked")

        self.declare_parameter("forward_step_m", DEFAULT_FORWARD_STEP_M)
        self.declare_parameter("turn_step_deg",  DEFAULT_TURN_STEP_DEG)
        self.declare_parameter("linear_x",       DEFAULT_LINEAR_X)
        self.declare_parameter("angular_z",      DEFAULT_ANGULAR_Z)

        self.declare_parameter("publish_rate_sec", DEFAULT_PUBLISH_RATE)
        self.declare_parameter("max_acc_linear",   DEFAULT_MAX_ACC_LIN)
        self.declare_parameter("max_acc_angular",  DEFAULT_MAX_ACC_ANG)

        def p(name):
            return self.get_parameter(name).value

        action_topic       = p("action_topic")
        cmd_vel_topic      = p("cmd_vel_topic")
        odom_topic         = p("odom_topic")
        status_topic       = p("status_topic")
        path_blocked_topic = p("path_blocked_topic")

        self._step_m   = p("forward_step_m")
        self._step_rad = math.radians(p("turn_step_deg"))
        self.lin       = p("linear_x")
        self.ang       = p("angular_z")

        self.max_acc_lin = p("max_acc_linear")
        self.max_acc_ang = p("max_acc_angular")
        publish_rate     = p("publish_rate_sec")

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._target_lin = 0.0
        self._target_ang = 0.0
        self._current_lin = 0.0
        self._current_ang = 0.0
        self._dt = publish_rate

        self._executing   = False     # a primitive is running
        self._start_pose  = None      # (x0, y0, yaw0) at primitive start
        self._prim_kind   = None      # "forward" | "turn"
        self._prim_target = 0.0       # meters (forward) or radians (turn)
        self._odom        = None
        self._deadline    = None
        self._path_blocked = False

        # ------------------------------------------------------------------
        # I/O
        # ------------------------------------------------------------------
        self.sub_action = self.create_subscription(String, action_topic, self._action_cb, 10)
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self._odom_cb,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1))
        self.sub_blocked = self.create_subscription(
            Bool, path_blocked_topic, self._blocked_cb, 10)

        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,                              # match twist_mux
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, qos_cmd_vel)
        self.pub_status  = self.create_publisher(String, status_topic, 10)

        self._publish_timer = self.create_timer(publish_rate, self._publish_cb)

        self.get_logger().info(
            f"action_node started\n"
            f"  action in : {action_topic}\n"
            f"  cmd out   : {cmd_vel_topic}\n"
            f"  odom      : {odom_topic}\n"
            f"  step      : forward={self._step_m}m turn={math.degrees(self._step_rad):.0f}deg\n"
            f"  vel       : lin={self.lin} m/s  ang={self.ang} rad/s"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _action_cb(self, msg: String):
        action = msg.data.strip().lower()
        if action == "stop":
            self._target_lin = self._target_ang = 0.0
            self._executing = False
            self._deadline = None
            self._publish_status("done")
        elif action == "forward":
            self._start_primitive("forward", self._step_m)
        elif action in ("left", "turn_left"):
            self._start_primitive("turn", self._step_rad)
        elif action in ("right", "turn_right"):
            self._start_primitive("turn", -self._step_rad)
        else:
            self.get_logger().warn(f"Unknown action '{msg.data}' -> done")
            self._publish_status("done")

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _blocked_cb(self, msg: Bool):
        self._path_blocked = msg.data

    def _publish_status(self, status: str):
        """status in {'done', 'aborted'}"""
        out = String()
        out.data = status
        self.pub_status.publish(out)

    # ------------------------------------------------------------------
    # Primitive lifecycle
    # ------------------------------------------------------------------
    def _start_primitive(self, kind: str, magnitude: float):
        if self._odom is None:
            self.get_logger().warn("No odom yet -> immediate done")
            self._publish_status("done")
            return

        x0, y0, yaw0 = self._pose_xy_yaw(self._odom)
        self._start_pose = (x0, y0, yaw0)
        self._prim_kind = kind
        tol = FORWARD_TOL_M if kind == "forward" else TURN_TOL_RAD
        self._prim_target = max(abs(magnitude) - tol, 0.0)
        if self._prim_target <= 0.0:
            self._publish_status("done")
            return

        if kind == "forward":
            self._target_lin, self._target_ang = self.lin, 0.0
        else:
            self._target_lin = 0.0
            self._target_ang = math.copysign(self.ang, magnitude)

        vel = self.lin if kind == "forward" else self.ang
        self._deadline = self.get_clock().now() + rclpy.duration.Duration(
            seconds=(self._prim_target / vel) * 3.0 + 1.0)
        self._executing = True   # last, after _deadline

    def _publish_cb(self):
        # safety: forward blocked -> abort, VLA re-plans from the new frame
        if self._executing and self._prim_kind == "forward" and self._path_blocked:
            self.get_logger().warn("PATH BLOCKED -> abort forward")
            self._target_lin = self._target_ang = 0.0
            self._executing = False
            self._deadline = None
            self._publish_status("aborted")

        if self._executing and self._odom is not None and self._deadline is not None:
            x, y, yaw = self._pose_xy_yaw(self._odom)
            x0, y0, yaw0 = self._start_pose
            if self._prim_kind == "forward":
                progress = math.hypot(x - x0, y - y0)
            else:
                dyaw = yaw - yaw0
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                progress = abs(dyaw)

            if self.get_clock().now() >= self._deadline:
                self.get_logger().warn("Primitive timeout -> forced stop + aborted")
                self._target_lin = self._target_ang = 0.0
                self._executing = False
                self._deadline = None
                self._publish_status("aborted")
            elif progress >= self._prim_target:
                self._target_lin = self._target_ang = 0.0
                self._executing = False
                self._deadline = None
                self._publish_status("done")

        self._current_lin = self._ramp(self._current_lin, self._target_lin, self.max_acc_lin * self._dt)
        self._current_ang = self._ramp(self._current_ang, self._target_ang, self.max_acc_ang * self._dt)
        twist = Twist()
        twist.linear.x, twist.angular.z = self._current_lin, self._current_ang
        self.pub_cmd_vel.publish(twist)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _ramp(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        delta = math.copysign(min(abs(delta), max_delta), delta)
        return current + delta

    @staticmethod
    def _pose_xy_yaw(odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return p.x, p.y, math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown: final STOP")
        node.pub_cmd_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()