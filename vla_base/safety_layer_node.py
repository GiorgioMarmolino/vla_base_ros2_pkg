#!/usr/bin/env python3
"""
safety_layer_node.py

Velocity guard for the NaVILA -> Husky pipeline.

    action_to_cmdvel_node --> /cmd_vel_raw --> [safety_layer_node] --> /cmd_vel --> twist_mux

Front-end: VLP-16 PointCloud2 -> base_link -> height-band ground removal ->
2D occupancy grid. From the grid it derives per-sector min distances (front/
left/right/rear) AND runs a predictive footprint sweep on the current command
(forward-simulated unicycle trajectory) to detect collisions the static cones
would miss (e.g. corners swinging in during a turn).

Velocity scaling keeps the exact same thresholds/formula as before
(front_stop_dist / front_slow_dist). On a hard block ahead it raises
`path_blocked` so the action node can self-abort the current primitive and let
NaVILA re-plan from the new frame (deviation via re-planning, not via steering).

Subscribes:
    /cmd_vel_raw      (geometry_msgs/Twist)      raw command from action node
    /velodyne_points  (sensor_msgs/PointCloud2)  VLP-16 cloud
    /zed/depth/depth_registered (Image)          optional, low front obstacles

Publishes:
    /cmd_vel              (geometry_msgs/Twist)  filtered/safe command
    /safety/path_blocked  (std_msgs/Bool)        hard block -> abort trigger
    /safety/occupancy     (nav_msgs/OccupancyGrid) optional debug grid
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy, qos_profile_sensor_data)

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid

from sensor_msgs_py import point_cloud2
import tf2_ros

from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_CLOUD_TIMEOUT   = 1.0    # s
DEFAULT_CMD_TIMEOUT     = 3.0    # s
DEFAULT_DEPTH_TIMEOUT   = 1.4    # s
DEFAULT_PUBLISH_RATE    = 0.05   # s  (20 Hz)

DEFAULT_MAX_ACC_LIN     = 1.0    # m/s^2
DEFAULT_MAX_ACC_ANG     = 2.0    # rad/s^2

DEFAULT_FRONT_STOP_DIST = 1.2    # m
DEFAULT_FRONT_SLOW_DIST = 1.5    # m
DEFAULT_SIDE_STOP_DIST  = 1.0    # m
DEFAULT_REAR_STOP_DIST  = 1.2    # m

DEFAULT_FRONT_FOV_DEG   = 60.0   # deg
DEFAULT_SIDE_FOV_DEG    = 15.0   # deg
DEFAULT_REAR_FOV_DEG    = 10.0   # deg

DEFAULT_TURN_REDUCTION  = 0.4

# Occupancy / height map
DEFAULT_BASE_FRAME      = "base_link"
DEFAULT_GROUND_CLIP_Z   = 0.10   # m  below this -> floor, ignored
DEFAULT_MAX_OBSTACLE_Z  = 1.50   # m  above this -> overhead clearance, ignored
DEFAULT_GRID_RES        = 0.05   # m
DEFAULT_GRID_RANGE      = 4.0    # m  half-size of the local grid (+/-)

# Robot footprint (Husky ~0.99 x 0.67 m, base_link centered)
DEFAULT_ROBOT_HALF_LEN  = 0.50   # m
DEFAULT_ROBOT_HALF_WID  = 0.34   # m
DEFAULT_FOOTPRINT_MARGIN = 0.12  # m

# Predictive sweep
DEFAULT_PREDICT_HORIZON = 2.0    # s
DEFAULT_PREDICT_DT      = 0.10   # s
DEFAULT_BLOCK_DEBOUNCE  = 3      # consecutive cycles before latching path_blocked

EPS_MOVE = 0.02


class SafetyLayerNode(Node):

    def __init__(self):
        super().__init__("safety_layer_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("cmd_in_topic",      "/cmd_vel_raw")
        self.declare_parameter("cmd_out_topic",     "/cmd_vel")
        self.declare_parameter("cloud_topic",       "/velodyne_points")
        self.declare_parameter("depth_topic",       "/zed/depth/depth_registered")
        self.declare_parameter("path_blocked_topic", "/safety/path_blocked")
        self.declare_parameter("enable_depth",      False)
        self.declare_parameter("publish_grid",      False)
        self.declare_parameter("occupancy_topic",   "/safety/occupancy")

        self.declare_parameter("base_frame",       DEFAULT_BASE_FRAME)
        self.declare_parameter("ground_clip_z",    DEFAULT_GROUND_CLIP_Z)
        self.declare_parameter("max_obstacle_z",   DEFAULT_MAX_OBSTACLE_Z)
        self.declare_parameter("grid_res",         DEFAULT_GRID_RES)
        self.declare_parameter("grid_range",       DEFAULT_GRID_RANGE)

        self.declare_parameter("robot_half_length", DEFAULT_ROBOT_HALF_LEN)
        self.declare_parameter("robot_half_width",  DEFAULT_ROBOT_HALF_WID)
        self.declare_parameter("footprint_margin",  DEFAULT_FOOTPRINT_MARGIN)
        self.declare_parameter("predict_horizon_sec", DEFAULT_PREDICT_HORIZON)
        self.declare_parameter("predict_dt",          DEFAULT_PREDICT_DT)
        self.declare_parameter("block_debounce",      DEFAULT_BLOCK_DEBOUNCE)

        self.declare_parameter("front_stop_dist", DEFAULT_FRONT_STOP_DIST)
        self.declare_parameter("front_slow_dist", DEFAULT_FRONT_SLOW_DIST)
        self.declare_parameter("side_stop_dist",  DEFAULT_SIDE_STOP_DIST)
        self.declare_parameter("rear_stop_dist",  DEFAULT_REAR_STOP_DIST)

        self.declare_parameter("front_fov_deg", DEFAULT_FRONT_FOV_DEG)
        self.declare_parameter("side_fov_deg",  DEFAULT_SIDE_FOV_DEG)
        self.declare_parameter("rear_fov_deg",  DEFAULT_REAR_FOV_DEG)

        self.declare_parameter("cloud_timeout_sec", DEFAULT_CLOUD_TIMEOUT)
        self.declare_parameter("cmd_timeout_sec",   DEFAULT_CMD_TIMEOUT)
        self.declare_parameter("depth_timeout_sec", DEFAULT_DEPTH_TIMEOUT)
        self.declare_parameter("publish_rate_sec",  DEFAULT_PUBLISH_RATE)
        self.declare_parameter("max_acc_linear",    DEFAULT_MAX_ACC_LIN)
        self.declare_parameter("max_acc_angular",   DEFAULT_MAX_ACC_ANG)
        self.declare_parameter("turn_reduction",    DEFAULT_TURN_REDUCTION)

        def p(name):
            return self.get_parameter(name).value

        cmd_in_topic        = p("cmd_in_topic")
        cmd_out_topic       = p("cmd_out_topic")
        cloud_topic         = p("cloud_topic")
        depth_topic         = p("depth_topic")
        path_blocked_topic  = p("path_blocked_topic")
        occupancy_topic     = p("occupancy_topic")
        self.enable_depth   = p("enable_depth")
        self.publish_grid   = p("publish_grid")

        self.base_frame      = p("base_frame")
        self.ground_clip_z   = p("ground_clip_z")
        self.max_obstacle_z  = p("max_obstacle_z")
        self.grid_res        = p("grid_res")
        self.grid_range      = p("grid_range")

        self.robot_half_len  = p("robot_half_length")
        self.robot_half_wid  = p("robot_half_width")
        self.footprint_margin = p("footprint_margin")
        self.predict_horizon = p("predict_horizon_sec")
        self.predict_dt      = p("predict_dt")
        self.block_debounce  = p("block_debounce")

        self.front_stop_dist = p("front_stop_dist")
        self.front_slow_dist = p("front_slow_dist")
        self.side_stop_dist  = p("side_stop_dist")
        self.rear_stop_dist  = p("rear_stop_dist")

        self.front_fov = math.radians(p("front_fov_deg"))
        self.side_fov  = math.radians(p("side_fov_deg"))
        self.rear_fov  = math.radians(p("rear_fov_deg"))

        self.cloud_timeout_sec = p("cloud_timeout_sec")
        self.cmd_timeout_sec   = p("cmd_timeout_sec")
        self.depth_timeout_sec = p("depth_timeout_sec")
        publish_rate           = p("publish_rate_sec")
        self.max_acc_lin       = p("max_acc_linear")
        self.max_acc_ang       = p("max_acc_angular")
        self.turn_reduction    = p("turn_reduction")

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._target_lin = 0.0
        self._target_ang = 0.0
        self._current_lin = 0.0
        self._current_ang = 0.0

        self._last_cmd_time   = self.get_clock().now()
        self._last_cloud_time = self.get_clock().now()
        self._last_depth_time = self.get_clock().now()

        # occupied cell centers in base_frame, shape (N, 2)
        self._cells = np.empty((0, 2), dtype=np.float32)

        self._front_min_dist = 999.0
        self._left_min_dist  = 999.0
        self._right_min_dist = 999.0
        self._rear_min_dist  = 999.0
        self._front_depth_dist = 999.0

        self._left_blocked  = False
        self._right_blocked = False
        self._rear_blocked  = False

        self._block_counter = 0
        self._path_blocked = False
        self._d_eff = 999.0

        self._dt = publish_rate
        self._bridge = CvBridge()

        # cached static transform cloud_frame -> base_frame (R, t)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._T_cache = {}

        # ------------------------------------------------------------------
        # I/O
        # ------------------------------------------------------------------
        self.sub_cmd   = self.create_subscription(Twist, cmd_in_topic, self._cmd_cb, 10)
        self.sub_cloud = self.create_subscription(
            PointCloud2, cloud_topic, self._cloud_cb, qos_profile_sensor_data)
        if self.enable_depth:
            self.sub_depth = self.create_subscription(
                Image, depth_topic, self._depth_cb, qos_profile_sensor_data)

        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_out_topic, qos_cmd_vel)
        self.pub_blocked = self.create_publisher(Bool, path_blocked_topic, 10)
        if self.publish_grid:
            self.pub_grid = self.create_publisher(OccupancyGrid, occupancy_topic, 1)

        self._publish_timer = self.create_timer(publish_rate, self._publish_cb)
        self._debug_timer   = self.create_timer(3.0, self._debug_cb)

        self.get_logger().info(
            f"safety_layer_node started\n"
            f"  cmd in/out : {cmd_in_topic} -> {cmd_out_topic}\n"
            f"  cloud      : {cloud_topic} (base={self.base_frame})\n"
            f"  depth      : {depth_topic if self.enable_depth else 'DISABLED'}\n"
            f"  blocked    : {path_blocked_topic}\n"
            f"  front      : stop={self.front_stop_dist}m slow={self.front_slow_dist}m\n"
            f"  footprint  : {2*self.robot_half_len:.2f}x{2*self.robot_half_wid:.2f}m "
            f"margin={self.footprint_margin}m\n"
            f"  ground band: [{self.ground_clip_z}, {self.max_obstacle_z}]m"
        )

    # ------------------------------------------------------------------
    # Input callbacks
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
            roi = depth[h // 2:h, w // 4:3 * w // 4]
            valid = roi[(roi > 0.1) & np.isfinite(roi)]
            # 2nd percentile instead of raw min -> robust to single-pixel speckle
            self._front_depth_dist = float(np.percentile(valid, 2)) if valid.size else 999.0
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}", throttle_duration_sec=2.0)

    def _cloud_cb(self, msg: PointCloud2):
        self._last_cloud_time = self.get_clock().now()

        xyz = self._cloud_to_xyz(msg)
        if xyz.size == 0:
            return

        T = self._lookup_transform(msg.header.frame_id)
        if T is None:
            return
        R, t = T
        pts = (xyz @ R.T) + t  # to base_frame

        # height-band ground removal
        z = pts[:, 2]
        mask = (z > self.ground_clip_z) & (z < self.max_obstacle_z)
        obs = pts[mask, :2]
        if obs.shape[0] == 0:
            self._cells = np.empty((0, 2), dtype=np.float32)
            self._reset_sectors()
            return

        # clip to local grid and rasterize (dedup -> bounded cell count)
        r = self.grid_range
        in_grid = (np.abs(obs[:, 0]) < r) & (np.abs(obs[:, 1]) < r)
        obs = obs[in_grid]
        if obs.shape[0] == 0:
            self._cells = np.empty((0, 2), dtype=np.float32)
            self._reset_sectors()
            return

        cell_idx = np.round(obs / self.grid_res).astype(np.int32)
        cell_idx = np.unique(cell_idx, axis=0)
        cells = cell_idx.astype(np.float32) * self.grid_res
        self._cells = cells

        self._update_sectors(cells)

        if self.publish_grid:
            self._publish_occupancy(cell_idx)

    # ------------------------------------------------------------------
    # Perception helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
        try:
            pts = point_cloud2.read_points_numpy(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            return np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        except AttributeError:
            gen = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            return np.array([[q[0], q[1], q[2]] for q in gen], dtype=np.float32)

    def _lookup_transform(self, cloud_frame: str):
        if cloud_frame in self._T_cache:
            return self._T_cache[cloud_frame]
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, cloud_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(
                f"TF {self.base_frame}<-{cloud_frame} not ready: {e}",
                throttle_duration_sec=2.0)
            return None
        tr = tf.transform.translation
        q = tf.transform.rotation
        R = self._quat_to_mat(q.x, q.y, q.z, q.w)
        t = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        self._T_cache[cloud_frame] = (R, t)  # static mount -> cache once
        return R, t

    @staticmethod
    def _quat_to_mat(x, y, z, w) -> np.ndarray:
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)

    def _reset_sectors(self):
        self._front_min_dist = 999.0
        self._left_min_dist = self._right_min_dist = self._rear_min_dist = 999.0
        self._left_blocked = self._right_blocked = self._rear_blocked = False

    def _update_sectors(self, cells: np.ndarray):
        # base_frame: front = +x, left = +y. No front offset needed.
        rng = np.hypot(cells[:, 0], cells[:, 1])
        bearing = np.arctan2(cells[:, 1], cells[:, 0])

        def sector_min(center, fov):
            d = np.abs(np.arctan2(np.sin(bearing - center), np.cos(bearing - center)))
            sel = rng[d <= fov / 2.0]
            return float(sel.min()) if sel.size else 999.0

        self._front_min_dist = sector_min(0.0,          self.front_fov)
        self._left_min_dist  = sector_min(math.pi / 2,  self.side_fov)
        self._right_min_dist = sector_min(-math.pi / 2, self.side_fov)
        self._rear_min_dist  = sector_min(math.pi,      self.rear_fov)

        self._left_blocked  = self._left_min_dist  < self.side_stop_dist
        self._right_blocked = self._right_min_dist < self.side_stop_dist
        self._rear_blocked  = self._rear_min_dist  < self.rear_stop_dist

    # ------------------------------------------------------------------
    # Predictive footprint sweep
    # ------------------------------------------------------------------
    def _predict_collision(self, v: float, w: float):
        """Forward-simulate the (v, w) command. Returns (collided, arc_length)."""
        cells = self._cells
        if cells.shape[0] == 0 or (abs(v) < EPS_MOVE and abs(w) < EPS_MOVE):
            return False, math.inf

        hl = self.robot_half_len + self.footprint_margin
        hw = self.robot_half_wid + self.footprint_margin
        n = int(self.predict_horizon / self.predict_dt)

        x = y = th = 0.0
        path = 0.0
        cx, cy = cells[:, 0], cells[:, 1]
        for _ in range(n):
            x += v * math.cos(th) * self.predict_dt
            y += v * math.sin(th) * self.predict_dt
            th += w * self.predict_dt
            path += abs(v) * self.predict_dt
            dx = cx - x
            dy = cy - y
            c, s = math.cos(th), math.sin(th)
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            if np.any((np.abs(lx) <= hl) & (np.abs(ly) <= hw)):
                return True, path
        return False, math.inf

    # ------------------------------------------------------------------
    # Main loop: timeout -> safety -> smoothing -> publish
    # ------------------------------------------------------------------
    def _publish_cb(self):
        now = self.get_clock().now()
        cloud_dt = (now - self._last_cloud_time).nanoseconds / 1e9
        cmd_dt   = (now - self._last_cmd_time).nanoseconds / 1e9

        lin = self._target_lin
        ang = self._target_ang
        blocked_now = False
        self._d_eff = 999.0

        if cmd_dt > self.cmd_timeout_sec:
            if lin != 0.0 or ang != 0.0:
                self.get_logger().warn(
                    f"CMD WATCHDOG: no command for {cmd_dt:.2f}s -> STOP",
                    throttle_duration_sec=1.0)
            lin = ang = 0.0

        elif cloud_dt > self.cloud_timeout_sec:
            self.get_logger().warn("CLOUD TIMEOUT -> STOP", throttle_duration_sec=1.0)
            lin = ang = 0.0

        else:
            # static front distance (lidar sector + fresh depth)
            front_static = self._front_min_dist
            if self.enable_depth:
                depth_dt = (now - self._last_depth_time).nanoseconds / 1e9
                if depth_dt <= self.depth_timeout_sec:
                    front_static = min(front_static, self._front_depth_dist)

            # predictive footprint sweep on the commanded trajectory
            collided, arclen = self._predict_collision(lin, ang)
            rot_block = collided and abs(lin) < EPS_MOVE
            d_pred = arclen if (collided and not rot_block) else math.inf

            d_eff = min(front_static, d_pred)
            self._d_eff = d_eff

            # forward protection (same thresholds/formula as before)
            if lin > 0.0:
                if d_eff < self.front_stop_dist:
                    self.get_logger().warn("OBSTACLE AHEAD -> STOP",
                                           throttle_duration_sec=1.0)
                    lin = 0.0
                    blocked_now = True
                elif d_eff < self.front_slow_dist:
                    scale = (d_eff - self.front_stop_dist) / \
                            (self.front_slow_dist - self.front_stop_dist)
                    lin *= max(0.0, min(scale, 1.0))

            # rear protection (reverse)
            if lin < 0.0 and self._rear_blocked:
                self.get_logger().warn("REAR OBSTACLE -> STOP", throttle_duration_sec=1.0)
                lin = 0.0

            # turn reduction near a side wall
            if lin > 0.0 and ang > 0.0 and self._left_blocked:
                self.get_logger().warn("LEFT SIDE CLOSE -> REDUCING TURN",
                                       throttle_duration_sec=1.0)
                ang *= self.turn_reduction
            if lin > 0.0 and ang < 0.0 and self._right_blocked:
                self.get_logger().warn("RIGHT SIDE CLOSE -> REDUCING TURN",
                                       throttle_duration_sec=1.0)
                ang *= self.turn_reduction

            # rotation that would sweep into an obstacle
            if rot_block:
                self.get_logger().warn("TURN WOULD HIT -> STOP ROTATION",
                                       throttle_duration_sec=1.0)
                ang = 0.0
                blocked_now = True

        self._update_blocked(blocked_now)

        # smoothing (acceleration ramp)
        self._current_lin = self._ramp(self._current_lin, lin, self.max_acc_lin * self._dt)
        self._current_ang = self._ramp(self._current_ang, ang, self.max_acc_ang * self._dt)

        twist = Twist()
        twist.linear.x = self._current_lin
        twist.angular.z = self._current_ang
        self.pub_cmd_vel.publish(twist)

    def _update_blocked(self, blocked_now: bool):
        if blocked_now:
            self._block_counter = min(self._block_counter + 1, self.block_debounce)
        else:
            self._block_counter = 0

        latched = self._block_counter >= self.block_debounce
        if latched != self._path_blocked:
            self._path_blocked = latched
            self.pub_blocked.publish(Bool(data=latched))
            self.get_logger().info(f"path_blocked -> {latched}")

    # ------------------------------------------------------------------
    # Debug grid
    # ------------------------------------------------------------------
    def _publish_occupancy(self, cell_idx: np.ndarray):
        size = int(2 * self.grid_range / self.grid_res)
        grid = np.zeros((size, size), dtype=np.int8)
        half = size // 2
        ij = cell_idx + half
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < size) & (ij[:, 1] >= 0) & (ij[:, 1] < size)
        ij = ij[ok]
        grid[ij[:, 1], ij[:, 0]] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.info.resolution = self.grid_res
        msg.info.width = size
        msg.info.height = size
        origin = Pose()
        origin.position.x = -self.grid_range
        origin.position.y = -self.grid_range
        msg.info.origin = origin
        msg.data = grid.flatten().tolist()
        self.pub_grid.publish(msg)

    def _debug_cb(self):
        now = self.get_clock().now()
        cloud_to = (now - self._last_cloud_time).nanoseconds / 1e9 > self.cloud_timeout_sec
        depth_str = "off"
        if self.enable_depth:
            depth_str = f"{self._front_depth_dist:.2f}m"
        self.get_logger().info(
            f"[SAFETY] cells={self._cells.shape[0]}  "
            f"front={self._front_min_dist:.2f}m depth={depth_str} d_eff={self._d_eff:.2f}m  "
            f"left={self._left_min_dist:.2f}m right={self._right_min_dist:.2f}m "
            f"rear={self._rear_min_dist:.2f}m  "
            f"blocked={self._path_blocked}  "
            f"| cloud_timeout={'YES' if cloud_to else 'ok'}  "
            f"target=({self._target_lin:.2f},{self._target_ang:.2f}) "
            f"out=({self._current_lin:.2f},{self._current_ang:.2f})"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _ramp(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        delta = math.copysign(min(abs(delta), max_delta), delta)
        return current + delta


def main(args=None):
    rclpy.init(args=args)
    node = SafetyLayerNode()
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