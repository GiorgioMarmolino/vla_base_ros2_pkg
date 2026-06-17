#!/usr/bin/env python3
"""
safety_layer_node.py

Safety layer del pipeline NaVILA -> Husky, estratto da action_to_cmdvel_node
e reso un nodo a parte.

Funziona come FILTRO DI VELOCITA' ("velocity guard"): prende il comando grezzo
prodotto dal nodo che traduce le azioni NaVILA, applica tutta la protezione
anti-collisione e ripubblica un comando "sicuro" verso twist_mux.

    action_to_cmdvel_node --> /cmd_vel_raw --> [safety_layer_node] --> /cmd_vel --> twist_mux

Subscribes:
    /cmd_vel_raw   (geometry_msgs/Twist)   comando grezzo dall'action node
    /scan          (sensor_msgs/LaserScan) LiDAR 2D (es. pointcloud_to_laserscan del VLP-16)
    /zed/depth/depth_registered (Image)    opzionale, ostacoli bassi davanti

Publishes:
    /cmd_vel       (geometry_msgs/Twist)   comando filtrato / sicuro

Sicurezze implementate:
    - Watchdog sul comando in ingresso (se l'action node muore -> STOP)
    - Timeout LiDAR fail-safe (se non arrivano scan -> STOP)
    - Protezione frontale: stop netto + rallentamento progressivo
    - Protezione posteriore (in retromarcia)
    - Riduzione della sterzata in corridoi stretti (lato vicino)
    - Depth camera per ostacoli bassi davanti (opzionale)
    - Smoothing finale con rampa di accelerazione
    - Logging throttled + timer di debug
    - Parametri ROS 2 completamente configurabili
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image

from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_LIDAR_TIMEOUT   = 5.0    # s   — max tempo senza scan
DEFAULT_CMD_TIMEOUT     = 3.0    # s   — watchdog sul comando in ingresso
DEFAULT_DEPTH_TIMEOUT   = 1.4    # s   — oltre questo la depth è considerata "stale"
DEFAULT_PUBLISH_RATE    = 0.05   # s   — periodo pubblicazione (20 Hz)

DEFAULT_MAX_ACC_LIN     = 1.0    # m/s²   — max accelerazione lineare
DEFAULT_MAX_ACC_ANG     = 2.0    # rad/s² — max accelerazione angolare

DEFAULT_FRONT_STOP_DIST = 1.2   # m   — distanza frontale per stop
DEFAULT_FRONT_SLOW_DIST = 1.5    # m   — distanza frontale per inizio rallentamento
DEFAULT_SIDE_STOP_DIST  = 1.0    # m   — distanza laterale
DEFAULT_REAR_STOP_DIST  = 1.2    # m   — distanza posteriore

DEFAULT_FRONT_FOV_DEG   = 60.0   # °
DEFAULT_SIDE_FOV_DEG    = 15.0   # °
DEFAULT_REAR_FOV_DEG    = 10.0   # °

# Offset fisico del "davanti" del robot nell'array dello scan.
# Per il Velodyne VLP-16 il fronte è ~90° (vedi anche idx = 3*n//4).
DEFAULT_LIDAR_FRONT_ANGLE_DEG = 0.0

DEFAULT_TURN_REDUCTION  = 0.4    # fattore di riduzione sterzata lato vicino


class SafetyLayerNode(Node):

    def __init__(self):
        super().__init__("safety_layer_node")

        # ------------------------------------------------------------------
        # Parametri ROS 2
        # ------------------------------------------------------------------
        self.declare_parameter("cmd_in_topic",  "/cmd_vel_raw")
        self.declare_parameter("cmd_out_topic", "/cmd_vel")
        self.declare_parameter("scan_topic",    "/scan")
        self.declare_parameter("depth_topic",   "/zed/depth/depth_registered")
        self.declare_parameter("enable_depth",  True)

        # Distanze di sicurezza
        self.declare_parameter("front_stop_dist", DEFAULT_FRONT_STOP_DIST)
        self.declare_parameter("front_slow_dist", DEFAULT_FRONT_SLOW_DIST)
        self.declare_parameter("side_stop_dist",  DEFAULT_SIDE_STOP_DIST)
        self.declare_parameter("rear_stop_dist",  DEFAULT_REAR_STOP_DIST)

        # FOV dei settori
        self.declare_parameter("front_fov_deg", DEFAULT_FRONT_FOV_DEG)
        self.declare_parameter("side_fov_deg",  DEFAULT_SIDE_FOV_DEG)
        self.declare_parameter("rear_fov_deg",  DEFAULT_REAR_FOV_DEG)

        # Offset del fronte nello scan
        self.declare_parameter("lidar_front_angle_deg", DEFAULT_LIDAR_FRONT_ANGLE_DEG)

        # Timeout / smoothing
        self.declare_parameter("lidar_timeout_sec", DEFAULT_LIDAR_TIMEOUT)
        self.declare_parameter("cmd_timeout_sec",   DEFAULT_CMD_TIMEOUT)
        self.declare_parameter("depth_timeout_sec", DEFAULT_DEPTH_TIMEOUT)
        self.declare_parameter("publish_rate_sec",  DEFAULT_PUBLISH_RATE)
        self.declare_parameter("max_acc_linear",    DEFAULT_MAX_ACC_LIN)
        self.declare_parameter("max_acc_angular",   DEFAULT_MAX_ACC_ANG)
        self.declare_parameter("turn_reduction",    DEFAULT_TURN_REDUCTION)

        def p(name):
            return self.get_parameter(name).value

        cmd_in_topic       = p("cmd_in_topic")
        cmd_out_topic      = p("cmd_out_topic")
        scan_topic         = p("scan_topic")
        depth_topic        = p("depth_topic")
        self.enable_depth  = p("enable_depth")

        self.front_stop_dist = p("front_stop_dist")
        self.front_slow_dist = p("front_slow_dist")
        self.side_stop_dist  = p("side_stop_dist")
        self.rear_stop_dist  = p("rear_stop_dist")

        self.front_fov_deg = p("front_fov_deg")
        self.side_fov_deg  = p("side_fov_deg")
        self.rear_fov_deg  = p("rear_fov_deg")

        self.lidar_front_angle_deg = p("lidar_front_angle_deg")

        self.lidar_timeout_sec = p("lidar_timeout_sec")
        self.cmd_timeout_sec   = p("cmd_timeout_sec")
        self.depth_timeout_sec = p("depth_timeout_sec")
        publish_rate           = p("publish_rate_sec")
        self.max_acc_lin       = p("max_acc_linear")
        self.max_acc_ang       = p("max_acc_angular")
        self.turn_reduction    = p("turn_reduction")

        # ------------------------------------------------------------------
        # Stato interno
        # ------------------------------------------------------------------
        self._target_lin: float  = 0.0   # comando grezzo in ingresso
        self._target_ang: float  = 0.0
        self._current_lin: float = 0.0   # comando smoothed corrente
        self._current_ang: float = 0.0

        self._last_cmd_time   = self.get_clock().now()
        self._last_scan_time  = self.get_clock().now()
        self._last_depth_time = self.get_clock().now()

        # Stato sicurezze
        self._front_min_dist = 999.0
        self._left_min_dist  = 999.0
        self._right_min_dist = 999.0
        self._rear_min_dist  = 999.0
        self._front_depth_dist = 999.0

        self._front_blocked = False
        self._left_blocked  = False
        self._right_blocked = False
        self._rear_blocked  = False

        self._dt = publish_rate
        self._bridge = CvBridge()

        # ------------------------------------------------------------------
        # Subscriber / Publisher / Timer
        # ------------------------------------------------------------------
        self.sub_cmd  = self.create_subscription(Twist, cmd_in_topic, self._cmd_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        if self.enable_depth:
            self.sub_depth = self.create_subscription(Image, depth_topic, self._depth_cb, qos_profile_sensor_data)

        # QoS allineato a twist_mux (come nel nodo originale)
        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_out_topic, qos_cmd_vel)

        self._publish_timer = self.create_timer(publish_rate, self._publish_cb)
        self._debug_timer   = self.create_timer(3.0, self._debug_cb)

        self.get_logger().info(
            f"safety_layer_node avviato\n"
            f"  cmd in    : {cmd_in_topic}\n"
            f"  cmd out   : {cmd_out_topic}\n"
            f"  scan      : {scan_topic}\n"
            f"  depth     : {depth_topic if self.enable_depth else 'DISABILITATA'}\n"
            f"  front     : stop={self.front_stop_dist}m slow={self.front_slow_dist}m fov={self.front_fov_deg}°\n"
            f"  side/rear : side={self.side_stop_dist}m rear={self.rear_stop_dist}m\n"
            f"  front_off : {self.lidar_front_angle_deg}°"
        )

    # ------------------------------------------------------------------
    # Callbacks input
    # ------------------------------------------------------------------
    def _cmd_cb(self, msg: Twist):
        self._target_lin = msg.linear.x
        self._target_ang = msg.angular.z
        self._last_cmd_time = self.get_clock().now()

    def _depth_cb(self, msg: Image):
        self._last_depth_time = self.get_clock().now()
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            h, w = depth.shape[:2]

            # ROI: zona centrale-bassa del frame (ostacoli vicini al suolo)
            roi = depth[h // 2:h, w // 4:3 * w // 4]

            # Filtra valori invalidi (0, inf, nan)
            valid = roi[(roi > 0.1) & np.isfinite(roi)]
            self._front_depth_dist = float(np.min(valid)) if valid.size > 0 else 999.0
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}", throttle_duration_sec=2.0)

    def _scan_cb(self, msg: LaserScan):
        """Settorizza il LaserScan in front / left / right / rear e calcola le
        distanze minime per ogni settore."""
        self._last_scan_time = self.get_clock().now()

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = ranges.size
        inc = msg.angle_increment
        if n == 0 or inc == 0.0:
            return

        # Indice del "davanti" fisico del robot.
        # NB: lidar_front_angle_deg è in GRADI, angle_min in RADIANTI -> converti!
        front_angle = math.radians(self.lidar_front_angle_deg)
        front_idx = int(round((front_angle - msg.angle_min) / inc)) % n
        left_idx  = (front_idx + n // 4) % n   # +90° (sinistra, +y in REP-103)
        rear_idx  = (front_idx + n // 2) % n   # 180°
        right_idx = (front_idx - n // 4) % n   # -90° (destra)

        rmin = msg.range_min if msg.range_min > 0.0 else 0.05
        rmax = msg.range_max if msg.range_max > 0.0 else 100.0

        def sector_min(center: int, fov_deg: float) -> float:
            # half = mezzo FOV in numero di campioni (il bug originale usava il FOV intero)
            half = max(1, int(math.radians(fov_deg) / 2.0 / inc))
            idxs = np.arange(center - half, center + half + 1) % n
            vals = ranges[idxs]
            vals = vals[np.isfinite(vals) & (vals >= rmin) & (vals <= rmax)]
            return float(vals.min()) if vals.size else 999.0

        self._front_min_dist = sector_min(front_idx, self.front_fov_deg)
        self._left_min_dist  = sector_min(left_idx,  self.side_fov_deg)
        self._right_min_dist = sector_min(right_idx, self.side_fov_deg)
        self._rear_min_dist  = sector_min(rear_idx,  self.rear_fov_deg)

        self._front_blocked = self._front_min_dist < self.front_slow_dist
        self._left_blocked  = self._left_min_dist  < self.side_stop_dist
        self._right_blocked = self._right_min_dist < self.side_stop_dist
        self._rear_blocked  = self._rear_min_dist  < self.rear_stop_dist

    # ------------------------------------------------------------------
    # Loop principale: timeout -> safety -> smoothing -> publish
    # ------------------------------------------------------------------
    def _publish_cb(self):
        now = self.get_clock().now()
        scan_dt = (now - self._last_scan_time).nanoseconds / 1e9
        cmd_dt  = (now - self._last_cmd_time).nanoseconds / 1e9

        lin = self._target_lin
        ang = self._target_ang

        # --- Watchdog comando: l'action node è morto / non pubblica più ---
        if cmd_dt > self.cmd_timeout_sec:
            if lin != 0.0 or ang != 0.0:
                self.get_logger().warn(
                    f"CMD WATCHDOG: nessun comando da {cmd_dt:.2f}s -> STOP",
                    throttle_duration_sec=1.0)
            lin = 0.0
            ang = 0.0

        # --- Timeout LiDAR: non mi fido a muovermi alla cieca ---
        elif scan_dt > self.lidar_timeout_sec:
            self.get_logger().warn("LIDAR TIMEOUT -> STOP", throttle_duration_sec=1.0)
            lin = 0.0
            ang = 0.0

        else:
            # Distanza frontale effettiva = min(lidar, depth se attiva e fresca)
            front_d = self._front_min_dist
            if self.enable_depth:
                depth_dt = (now - self._last_depth_time).nanoseconds / 1e9
                if depth_dt <= self.depth_timeout_sec:
                    front_d = min(front_d, self._front_depth_dist)

            # Protezione frontale (solo se sto andando avanti)
            if lin > 0.0:
                if front_d < self.front_stop_dist:          # Hard stop
                    self.get_logger().warn("FRONT OBSTACLE -> STOP", throttle_duration_sec=1.0)
                    lin = 0.0
                elif front_d < self.front_slow_dist:        # Rallentamento progressivo
                    scale = (front_d - self.front_stop_dist) / \
                            (self.front_slow_dist - self.front_stop_dist)
                    lin *= max(0.0, min(scale, 1.0))

            # Protezione posteriore (in retromarcia)
            if lin < 0.0 and self._rear_blocked:
                self.get_logger().warn("REAR OBSTACLE -> STOP", throttle_duration_sec=1.0)
                lin = 0.0

            # Riduzione sterzata in corridoi stretti (lato vicino)
            if lin > 0.0 and ang > 0.0 and self._left_blocked:
                self.get_logger().warn("LEFT SIDE CLOSE -> REDUCING TURN", throttle_duration_sec=1.0)
                ang *= self.turn_reduction
            if lin > 0.0 and ang < 0.0 and self._right_blocked:
                self.get_logger().warn("RIGHT SIDE CLOSE -> REDUCING TURN", throttle_duration_sec=1.0)
                ang *= self.turn_reduction

        # --- Smoothing finale (rampa di accelerazione) ---
        self._current_lin = self._ramp(self._current_lin, lin, self.max_acc_lin * self._dt)
        self._current_ang = self._ramp(self._current_ang, ang, self.max_acc_ang * self._dt)

        twist = Twist()
        twist.linear.x  = self._current_lin
        twist.angular.z = self._current_ang
        self.pub_cmd_vel.publish(twist)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _debug_cb(self):
        now = self.get_clock().now()
        lidar_to = (now - self._last_scan_time).nanoseconds / 1e9 > self.lidar_timeout_sec
        depth_str = "off"
        if self.enable_depth:
            depth_block = self._front_depth_dist < self.front_slow_dist
            depth_str = f"{self._front_depth_dist:.2f}m ({'BLOCK' if depth_block else 'ok'})"

        self.get_logger().info(
            f"[SAFETY] "
            f"front_lidar={self._front_min_dist:.2f}m ({'BLOCK' if self._front_blocked else 'ok'})  "
            f"front_depth={depth_str}  "
            f"left={self._left_min_dist:.2f}m ({'BLOCK' if self._left_blocked else 'ok'})  "
            f"right={self._right_min_dist:.2f}m ({'BLOCK' if self._right_blocked else 'ok'})  "
            f"rear={self._rear_min_dist:.2f}m ({'BLOCK' if self._rear_blocked else 'ok'})  "
            f"| lidar_timeout={'YES' if lidar_to else 'ok'}  "
            f"| target=({self._target_lin:.2f}, {self._target_ang:.2f}) "
            f"out=({self._current_lin:.2f}, {self._current_ang:.2f})"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _ramp(current: float, target: float, max_delta: float) -> float:
        """Avvicina current a target di al massimo max_delta."""
        delta = target - current
        delta = math.copysign(min(abs(delta), max_delta), delta)
        return current + delta


# =============================================================================
# Entry point
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SafetyLayerNode()
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