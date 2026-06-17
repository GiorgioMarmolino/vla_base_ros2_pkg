#!/usr/bin/env python3
"""
action_node.py

Converte le azioni NaVILA in comandi di velocità (geometry_msgs/Twist).

Il safety layer (collision avoidance, depth, LiDAR) è stato spostato in un nodo
a parte (safety_layer_node). Questo nodo si occupa SOLO di:
    - tradurre azioni/JSON in velocità target
    - watchdog sul comando (STOP se nessuna azione)
    - smoothing con rampa di accelerazione
    - pubblicare a rate fisso

Il topic di uscita è parametrico (`cmd_vel_topic`) così dal launch file puoi
deciderlo in base al flag della safety:
    - safety ON  -> /cmd_vel_raw  (lo prende il safety node)
    - safety OFF -> /cmd_vel      (va diretto a twist_mux)

Subscribes:
    /navila/action (std_msgs/String)
        1) JSON:   {"linear_x": 0.3, "angular_z": 0.2}
        2) Token:  forward | forward_fast | backward | turn_left | turn_right |
                   curve_left | curve_right | left | right | stop

Publishes:
    <cmd_vel_topic> (geometry_msgs/Twist)
"""

# import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry



# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_LINEAR_X       = 0.4    # m/s    — "forward"
DEFAULT_LINEAR_X_FAST  = 0.7    # m/s    — "forward_fast" NON UTILIZZATO
DEFAULT_LINEAR_X_BACK  = 0.2    # m/s    — "backward" NON UTILIZZATO
DEFAULT_ANGULAR_Z      = 0.35   # rad/s  — rotazione sul posto
DEFAULT_CURVE_LINEAR   = 0.2    # m/s    — componente lineare in "curve_*" NON UTILIZZATO
DEFAULT_CURVE_ANGULAR  = 0.3    # rad/s  — componente angolare in "curve_*" NON UTILIZZATO

DEFAULT_CMD_TIMEOUT    = 1.0    # s      — watchdog: stop se nessun cmd NON UTILIZZATO
DEFAULT_WATCHDOG_RATE  = 0.05   # s      — periodo timer watchdog (20 Hz) NON UTILIZZATO
DEFAULT_PUBLISH_RATE   = 0.05   # s      — periodo pubblicazione (20 Hz)

DEFAULT_MAX_ACC_LIN    = 1.0    # m/s²   — max accelerazione lineare
DEFAULT_MAX_ACC_ANG    = 2.0    # rad/s² — max accelerazione angolare


class ActionNode(Node):

    def __init__(self):
        super().__init__("action_node")

        # ------------------------------------------------------------------
        # Parametri ROS 2
        # ------------------------------------------------------------------
        self.declare_parameter("action_topic",  "/navila/action")
        self.declare_parameter("cmd_vel_topic",  "/cmd_vel")
        self.declare_parameter("odom_topic", "/platform/odom/filtered")
        self.declare_parameter("status_topic", "/navila/primitive_status")


        self.declare_parameter("linear_x",       DEFAULT_LINEAR_X)
        self.declare_parameter("linear_x_fast",  DEFAULT_LINEAR_X_FAST)
        self.declare_parameter("linear_x_back",  DEFAULT_LINEAR_X_BACK)
        self.declare_parameter("angular_z",      DEFAULT_ANGULAR_Z)
        self.declare_parameter("curve_linear",   DEFAULT_CURVE_LINEAR)
        self.declare_parameter("curve_angular",  DEFAULT_CURVE_ANGULAR)

        self.declare_parameter("cmd_timeout_sec",   DEFAULT_CMD_TIMEOUT)
        self.declare_parameter("watchdog_rate_sec", DEFAULT_WATCHDOG_RATE)
        self.declare_parameter("publish_rate_sec",  DEFAULT_PUBLISH_RATE)
        self.declare_parameter("max_acc_linear",    DEFAULT_MAX_ACC_LIN)
        self.declare_parameter("max_acc_angular",   DEFAULT_MAX_ACC_ANG)

        def p(name):
            return self.get_parameter(name).value

        action_topic  = p("action_topic")
        cmd_vel_topic = p("cmd_vel_topic")
        odom_topic    = p("odom_topic")
        status_topic  = p("status_topic")

        self.lin       = p("linear_x")
        self.lin_fast  = p("linear_x_fast")
        self.lin_back  = p("linear_x_back")
        self.ang       = p("angular_z")
        self.curve_lin = p("curve_linear")
        self.curve_ang = p("curve_angular")

        self.timeout_sec = p("cmd_timeout_sec")
        self.max_acc_lin = p("max_acc_linear")
        self.max_acc_ang = p("max_acc_angular")
        watchdog_rate    = p("watchdog_rate_sec")
        publish_rate     = p("publish_rate_sec")

        # ------------------------------------------------------------------
        # Mappa token → (linear_x, angular_z)
        # Estendibile senza toccare la logica del callback
        # ------------------------------------------------------------------
        # self._action_map: dict[str, tuple[float, float]] = {
        #     "forward":      ( self.lin,       0.0),
        #     "forward_fast": ( self.lin_fast,  0.0),
        #     "backward":     (-self.lin_back,  0.0),
        #     "turn_left":    ( 0.0,            self.ang),
        #     "turn_right":   ( 0.0,           -self.ang),
        #     "curve_left":   ( self.curve_lin, self.curve_ang),
        #     "curve_right":  ( self.curve_lin,-self.curve_ang),
        #     # alias brevi per compatibilità con nodi legacy
        #     "left":         ( 0.0,            self.ang),
        #     "right":        ( 0.0,           -self.ang),
        #     "stop":         ( 0.0,            0.0),
        # }

        # ------------------------------------------------------------------
        # Stato interno
        # ------------------------------------------------------------------
        self._target_lin: float  = 0.0   # velocità target (da azione)
        self._target_ang: float  = 0.0
        self._current_lin: float = 0.0   # velocità smoothed corrente
        self._current_ang: float = 0.0
        # self._last_cmd_time = self.get_clock().now()
        self._dt = publish_rate          # usato per la rampa di accelerazione

        self._executing   = False    # True mentre una primitiva è in esecuzione
        self._start_pose  = None     # (x0, y0, yaw0) all'avvio della primitiva
        self._prim_kind   = None     # "forward" | "turn"
        self._prim_target = 0.0      # target: metri (forward) o radianti (turn)
        self._odom        = None     # ultimo Odometry ricevuto

        self._deadline = None     # deadline per completare la primitiva (ora + tempo massimo)

        # ------------------------------------------------------------------
        # Subscriber / Publisher / Timer
        # ------------------------------------------------------------------
        self.sub_action = self.create_subscription(String, action_topic, self._action_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self._odom_cb,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, 
                depth=1))
            
        qos_cmd_vel = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,                              # match twist_mux
                durability=DurabilityPolicy.VOLATILE,
            )
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, qos_cmd_vel)
        self.pub_status  = self.create_publisher(String, status_topic, 10)

        # self._watchdog_timer = self.create_timer(watchdog_rate, self._watchdog_cb)
        self._publish_timer  = self.create_timer(publish_rate,  self._publish_cb)

        self.get_logger().info(
            f"action_node avviato\n"
            f"  topic in  : {action_topic}\n"
            f"  topic out : {cmd_vel_topic}\n"
            f"  odom      : {odom_topic}\n"
            f"  vel       : lin={self.lin} m/s  ang={self.ang} rad/s\n"
            f"  max_acc   : lin={self.max_acc_lin} m/s²  ang={self.max_acc_ang} rad/s²"
        )

    # ------------------------------------------------------------------
    # Action callback
    # ------------------------------------------------------------------
    # def _action_cb(self, msg: String):
    #     raw = msg.data.strip()

    #     # --- Parse as JSON -----------------------------------------------
    #     if raw.startswith("{"):
    #         try:
    #             data = json.loads(raw)
    #             lx = float(data.get("linear_x",  0.0))
    #             az = float(data.get("angular_z", 0.0))
    #             self._set_target(lx, az, label=f"JSON({lx:.2f},{az:.2f})")
    #             return
    #         except (json.JSONDecodeError, ValueError) as exc:
    #             self.get_logger().warn(f"JSON malformato: '{raw}' — {exc} → STOP")
    #             self._set_target(0.0, 0.0, label="JSON_ERROR→stop")
    #             return

    #     # --- Token string ------------------------------------------------
    #     action = raw.lower()
    #     if action in self._action_map:
    #         lx, az = self._action_map[action]
    #         self._set_target(lx, az, label=action)
    #     else:
    #         self.get_logger().warn(f"Azione non riconosciuta: '{action}' → STOP")
    #         self._set_target(0.0, 0.0, label="unknown→stop")

    def _action_cb(self, msg: String):
        parts = msg.data.strip().lower().split()
        if not parts:
            return
        action = parts[0]
        value  = float(parts[1]) if len(parts) > 1 else 0.0
        unit   = parts[2] if len(parts) > 2 else ""

        if action == "stop":
            self._publish_status("done")        # chiude comunque il ciclo del loop
            return
        if action == "forward" and unit == "cm":
            self._start_primitive(kind="forward", magnitude=value / 100.0)   # m
        elif action in ("turn_left", "turn_right") and unit == "deg":
            sign = +1.0 if action == "turn_left" else -1.0
            self._start_primitive(kind="turn", magnitude=sign * math.radians(value))  # rad
        else:
            self.get_logger().warn(f"Azione non gestita: '{msg.data}' → done")
            self._publish_status("done")

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _publish_status(self, status: str):
        """status ∈ {'done', 'aborted'}"""
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

    # def _set_target(self, lx: float, az: float, label: str = ""):
    #     self._target_lin = lx
    #     self._target_ang = az
    #     self._last_cmd_time = self.get_clock().now()
    #     self.get_logger().debug(f"target set [{label}] → lin={lx:.3f}  ang={az:.3f}")

    def _start_primitive(self, kind: str, magnitude: float):
        if self._odom is None:
            self.get_logger().warn("Nessun odom ancora → done immediato")
            self._publish_status("done")
            return
        x0, y0, yaw0 = self._pose_xy_yaw(self._odom)
        self._start_pose = (x0, y0, yaw0)
        self._prim_kind = kind
        self._prim_target = max(abs(magnitude) - (0.01 if kind == "forward" else math.radians(1.0)), 0.0)
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
        self._executing = True   # ← ultimo, dopo _deadline

    # ------------------------------------------------------------------
    # Watchdog callback
    # ------------------------------------------------------------------
    # def _watchdog_cb(self):
    #     dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
    #     if dt > self.timeout_sec:
    #         if self._target_lin != 0.0 or self._target_ang != 0.0:
    #             self.get_logger().info(f"Watchdog: nessun comando da {dt:.2f}s → STOP")
    #         self._target_lin = 0.0
    #         self._target_ang = 0.0

    # ------------------------------------------------------------------
    # Publish callback (smoothing + pubblicazione)
    # ------------------------------------------------------------------
    # def _publish_cb(self):
    #     # Smoothing con rampa di accelerazione verso il target
    #     self._current_lin = self._ramp(self._current_lin, self._target_lin, self.max_acc_lin * self._dt)
    #     self._current_ang = self._ramp(self._current_ang, self._target_ang, self.max_acc_ang * self._dt)

    #     twist = Twist()
    #     twist.linear.x  = self._current_lin
    #     twist.angular.z = self._current_ang
    #     self.pub_cmd_vel.publish(twist)

    #     self.get_logger().debug(
    #         f"cmd_vel -> lin={self._current_lin:.3f} ang={self._current_ang:.3f}")
    def _publish_cb(self):
        if self._executing and self._odom is not None and self._deadline is not None:
            x, y, yaw = self._pose_xy_yaw(self._odom)
            x0, y0, yaw0 = self._start_pose
            if self._prim_kind == "forward":
                progress = math.hypot(x - x0, y - y0)
            else:
                dyaw = yaw - yaw0
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # normalizza
                progress = abs(dyaw)

            if self.get_clock().now() >= self._deadline:
                self.get_logger().warn("Primitiva in timeout → stop forzato + aborted")
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
        """Avvicina current a target di al massimo max_delta."""
        delta = target - current
        delta = math.copysign(min(abs(delta), max_delta), delta)
        return current + delta

    @staticmethod
    def _pose_xy_yaw(odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        # yaw da quaternione (z-up)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return p.x, p.y, yaw

# =============================================================================
# Entry point
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown: invio STOP finale")
        stop = Twist()
        node.pub_cmd_vel.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()