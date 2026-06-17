#!/usr/bin/env python3
"""
docstring da riscrivere
"""

# =============================================================================
# Mock deepspeed — required only for training, not inference.
# Avoids import errors on environments without the full CUDA dev toolkit.
# =============================================================================
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

_mock_ds = MagicMock()
_mock_ds.__spec__ = "deepspeed"
_mock_ds.__version__ = "0.0.0"
for _mod in [
    "deepspeed",
    "deepspeed.comm",
    "deepspeed.runtime",
    "deepspeed.runtime.zero",
    "deepspeed.runtime.zero.partition_parameters",
    "deepspeed.runtime.activation_checkpointing",
    "deepspeed.runtime.activation_checkpointing.checkpointing",
]:
    sys.modules[_mod] = _mock_ds
# =============================================================================

import os
import re
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Empty

from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage

from collections import deque



# Official patterns
_OFFICIAL_PATTERNS = {
    "stop":       re.compile(r"\bstop\b", re.IGNORECASE),
    "forward":    re.compile(r"\bis move forward\b", re.IGNORECASE),
    "turn_left":  re.compile(r"\bis turn left\b", re.IGNORECASE),
    "turn_right": re.compile(r"\bis turn right\b", re.IGNORECASE),
}





def parse_navila_output(text: str):
    """Ritorna (action, value, unit) come da repo ufficiale.
    action ∈ {stop, forward, turn_left, turn_right}."""
    action = None
    for name, pat in _OFFICIAL_PATTERNS.items():
        if pat.search(text):
            action = name
            break
    if action is None:
        action = "stop"   # default ufficiale

    if action == "forward":
        m = re.search(r"move forward (\d+) cm", text)
        d = int(m.group(1)) if m else 25
        if d % 25 != 0:
            d = min([25, 50, 75], key=lambda x: abs(x - d))
        return "forward", d, "cm"
    if action == "turn_left":
        m = re.search(r"turn left (\d+) degree", text)
        g = int(m.group(1)) if m else 15
        if g % 15 != 0:
            g = min([15, 30, 45], key=lambda x: abs(x - g))
        return "turn_left", g, "deg"
    if action == "turn_right":
        m = re.search(r"turn right (\d+) degree", text)
        g = int(m.group(1)) if m else 15
        if g % 15 != 0:
            g = min([15, 30, 45], key=lambda x: abs(x - g))
        return "turn_right", g, "deg"
    return "stop", 0, ""




# =============================================================================
# NaVILA model loader
# =============================================================================

def load_navila_model(model_path: str):
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from huggingface_hub import snapshot_download

    HF_MODEL_ID = "a8cheng/navila-llama3-8b-8f"

    torch.cuda.empty_cache()

    # DS_ACCELERATOR=cpu affects only DeepSpeed (mocked); PyTorch uses CUDA normally.
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    os.environ["DS_ACCELERATOR"]     = "cpu"

    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"[NaVILA] Downloading model from HuggingFace: {HF_MODEL_ID}")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        print(f"[NaVILA] Model saved to: {model_path}")
    else:
        print(f"[NaVILA] Model found at: {model_path}")

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto",
        offload_folder="offload",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    llm = getattr(model, "llm", model)
    print("attn impl:", llm.config._attn_implementation)

    print(f"[NaVILA] Model device: {next(model.parameters()).device}")
    print("[NaVILA] Model loaded successfully.")
    return model, tokenizer, image_processor


# =============================================================================
# NaVILA inference
# =============================================================================

def run_navila_inference(
    model,
    tokenizer,
    image_processor,
    frames_rgb: list,
    goal: str,
    num_video_frames: int,
) -> str:
    """
    Un passo di inferenza NaVILA su una sequenza di frame (memoria + osservazione corrente).
    Ritorna il testo grezzo del modello (es. "the next action is to move forward 75 cm").
    """
    import torch
    from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates, SeparatorStyle

    # Lista di PIL → tensore impilato (N, C, H, W). N DEVE essere == num_video_frames.
    pil_images = [PILImage.fromarray(f) for f in frames_rgb]
    image_tensor = process_images(pil_images, image_processor, model.config)
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")

    conv = conv_templates["llama_3"].copy()
    image_token = "<image>\n"

    # Prompt ufficiale NaVILA: storico + osservazione corrente.
    qs = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {image_token * (num_video_frames - 1)}, and current observation <image>\n. "
        f"Your assigned task is: \"{goal}\" "
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=32,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stopping_criteria],
        )

    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
    return raw_output


# =============================================================================
# ROS 2 Node
# =============================================================================

class NaViLANode(Node):

    def __init__(self):
        super().__init__("navila_super_node")

        # ------------------------------------------------------------------
        # ROS 2 parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_path", os.environ.get("NAVILA_MODEL_PATH", "/models"))
        self.declare_parameter("phi3_model_path", "/models/phi3mini")                           # NON UTILIZZATO
        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("num_video_frames", 8) # default N=8 frames (7 historical + 1 current) as per NaVILA paper
        self.declare_parameter("max_history_frames", 512) # memory
        self.declare_parameter("frame_wait_timeout_sec", 1.0)  # attesa max frame post-moto
        self.declare_parameter("frame_settle_sec", 0.0)        # margine settling robot/camera
        self.declare_parameter("input_color_order", "bgr")

        self.declare_parameter("image_topic",  "/zed/rgb/color/rect/image/compressed")
        self.declare_parameter("goal_topic",   "/goal_instruction")
        self.declare_parameter("odom_topic",   "/platform/odom")

        self.declare_parameter("action_topic", "/navila/action")
        self.declare_parameter("reset_topic",  "/navila/reset")
        self.declare_parameter("status_topic",   "/navila/primitive_status")

        self.declare_parameter("use_phi3",     True)   # enable Phi-3 classifier                # NON UTILIZZATO
        self.declare_parameter("phi3_4bit",    False)    # quantize Phi-3 to 4-bit              # NON UTILIZZATO

        def p(name):
            return self.get_parameter(name).value

        model_path        = p("model_path")
        # inference_rate_hz = p("inference_rate_hz")                                              # NON UTILIZZATO
        self._num_video_frames = p("num_video_frames")
        max_history_frames = p("max_history_frames")
        self._frame_wait_timeout = p("frame_wait_timeout_sec")
        self._frame_settle       = p("frame_settle_sec")


        self._input_color_order = str(p("input_color_order")).strip().lower()
        if self._input_color_order not in ("bgr", "rgb"):
            self.get_logger().warn(f"input_color_order='{self._input_color_order}' not validido → use 'bgr'")
            self._input_color_order = "bgr"
        self.get_logger().info(f"input_color_order = {self._input_color_order}")

        image_topic       = p("image_topic")
        goal_topic        = p("goal_topic")
        # odom_topic        = p("odom_topic")                                                     # NON UTILIZZATO
        action_topic      = p("action_topic")
        reset_topic       = p("reset_topic")
        status_topic        = p("status_topic")

        # self._use_phi3    = p("use_phi3")                                                       # NON UTILIZZATO
        # self._phi3_4bit   = p("phi3_4bit")                                                      # NON UTILIZZATO        

        self._inference_running = False


        self.get_logger().info(f"use_phi3 = {self.get_parameter('use_phi3').value}")

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self.bridge        = CvBridge()
        self.last_frame    = None       # numpy RGB
        self._last_image_msg = None 
        self._frame_history = deque(maxlen=max_history_frames)
        self._frame_seq        = 0
        self._motion_done_seq  = 0
        self._motion_done_mono = time.monotonic()

        self._last_decision_frame   = None
        self._queue                 = []
        self._active                = False
        self._cycle_active          = False

        self.last_goal     = ""
        self.model         = None
        self.tokenizer     = None
        self.image_proc    = None
        self._model_ready  = False
        self._lock         = threading.Lock()

        # ------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------]]--
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_image = self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_cb,
            qos_sensor)

        self.sub_goal  = self.create_subscription(
            String, 
            goal_topic, 
            self._goal_cb, 10)
        
        self.sub_reset = self.create_subscription(
            Empty,
            reset_topic,
            self._reset_cb, 10)
        
        self.sub_status = self.create_subscription(
            String, 
            status_topic, 
            self._primitive_status_cb, 10)

        # ------------------------------------------------------------------
        # Publisher
        # ------------------------------------------------------------------
        self.pub_action = self.create_publisher(String, action_topic, 10)

        # ------------------------------------------------------------------
        # Inference timer
        # ------------------------------------------------------------------
        self.timer = self.create_timer(0.5, self._kick_drive)

        # ------------------------------------------------------------------
        # Load models in a background thread (non-blocking for ROS 2 spin)
        # ------------------------------------------------------------------
        self.get_logger().info(f"Loading NaVILA from: {model_path}")
        threading.Thread(
            target=self._load_model_thread,
            args=(model_path,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------
    def _image_cb(self, msg: CompressedImage):
        with self._lock:
            self._last_image_msg = msg
            self._frame_seq += 1

    def _process_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   # BGR diretto
            if frame is None:
                raise ValueError("cv2.imdecode returned None — corrupted frame?")
            if self._input_color_order == "bgr":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as exc:
            self.get_logger().warn(f"Image conversion error: {exc}")
            return None

    def _primitive_status_cb(self, msg: String):
        status = msg.data.strip().lower()
        with self._lock:
            if status in ("done", "aborted"):
                self._motion_done_seq = self._frame_seq
                self._motion_done_mono = time.monotonic()
            if status == "aborted":
                self._queue = []                  # coda invalidata dall'ostacolo
                self._last_decision_frame = None  # moto non avvenuto → fuori dallo storico
                self.get_logger().warn("Primitiva ABORTED → coda svuotata, frame scartato")
            # su 'done' non tocco coda né _last_decision_frame:
            #   il frame verrà promosso nello storico dal prossimo _drive_thread
            self._cycle_active = False
        self._kick_drive()

    def _goal_cb(self, msg: String):
        with self._lock:
            self.last_goal = msg.data
            self._frame_history.clear()
            self._last_decision_frame = None
            self._queue = []
            self._active = True
            self._cycle_active = False
            self._motion_done_seq  = self._frame_seq
            self._motion_done_mono = time.monotonic()
        self.get_logger().info(f"New goal: '{msg.data}' (loop armed)")
        self._kick_drive()

    def _reset_cb(self, msg: Empty):
        with self._lock:
            self.last_goal = ""
            self._active = False
            self._cycle_active = False
            self._frame_history.clear()
            self._last_decision_frame = None
            self._queue = []
        self.get_logger().info("NaVILA reset (loop disarmed).")

    # ------------------------------------------------------------------
    # Model loading (background thread)
    # ------------------------------------------------------------------

    def _load_model_thread(self, model_path: str):
        try:
            # <--- NaVILA --->
            self.get_logger().info("Loading NaVILA model...")
            model, tokenizer, image_proc = load_navila_model(model_path)
            self.get_logger().info("NaVILA model loaded successfully.")

            cfg_nvf = getattr(model.config, "num_video_frames", None)
            # --- Ready ---
            with self._lock:
                self.model        = model
                self.tokenizer    = tokenizer
                self.image_proc   = image_proc
                
                if cfg_nvf is None:
                    self.get_logger().warn(f"model.config.num_video_frames missing — keep ROS param ({self._num_video_frames}).")
                elif cfg_nvf != self._num_video_frames:
                    self.get_logger().warn(f"num_video_frames: param={self._num_video_frames} - checkpoint={cfg_nvf} → using checkpoint.")
                    self._num_video_frames = cfg_nvf
                else:
                    self._num_video_frames = cfg_nvf
                self.get_logger().info(f"num_video_frames correct = {self._num_video_frames}")
                self._model_ready = True

            self.get_logger().info(f"NaVILA ready — action parser: REGEX PARSER")

        except Exception as exc:
            self.get_logger().error(f"Failed to load NaVILA model: {exc}")

    # ------------------------------------------------------------------
    # Inference callback 09 / 06 / 2026 - versione con padding + debug
    # ------------------------------------------------------------------
    def _kick_drive(self):
        with self._lock:
            if self._cycle_active:
                return
            if not (self._model_ready and self._active and self._last_image_msg is not None):
                return
            self._cycle_active = True
        threading.Thread(target=self._drive_thread, daemon=True).start()

    def _drive_thread(self):
        try:
            with self._lock:
                # La storia avanza a ogni primitiva (eseguita): append del frame della
                # primitiva precedente, poi grab del corrente.
                if self._last_decision_frame is not None:
                    self._frame_history.append(self._last_decision_frame)
                    self._last_decision_frame = None
                goal      = self.last_goal
                model, tok, iproc = self.model, self.tokenizer, self.image_proc
                queued = self._queue.pop(0) if self._queue else None
                after_seq  = self._motion_done_seq
                after_mono = self._motion_done_mono
                settle     = self._frame_settle
                timeout    = self._frame_wait_timeout

            image_msg, stale = self._wait_fresh_frame(after_seq, after_mono, settle, timeout)
            if stale:
                self.get_logger().warn(
                    "Nessun frame fresco entro il timeout — uso l'ultimo disponibile.",
                    throttle_duration_sec=5.0)

            curr = self._process_image(image_msg)
            if curr is None:
                with self._lock:
                    self._cycle_active = False
                return

            # --- REPLAY: coda non vuota → esegui primitiva accodata, NIENTE inferenza ---
            if queued is not None:
                cmd = queued
                with self._lock:
                    self._last_decision_frame = curr
                self.get_logger().info(
                    f"[queue] {cmd}  (resto coda:{len(self._queue)}, hist:{len(self._frame_history)})")
                self._save_debug_frame(curr, cmd, "[queued]", goal)
                out = String(); out.data = cmd
                self.pub_action.publish(out)
                return   # resta in volo fino al prossimo primitive_done

            # --- DECISIONE: coda vuota → una inferenza, espandi, esegui 1, accoda il resto ---
            frames = self._sample_history(list(self._frame_history) + [curr], self._num_video_frames)
            raw_output = run_navila_inference(model, tok, iproc, frames, goal, self._num_video_frames)
            action, value, unit = parse_navila_output(raw_output)
            cmd, n_total = self._expand_primitives(action, value)

            if action == "stop":
                with self._lock:
                    self._last_decision_frame = curr
                    self._active = False
                    self._cycle_active = False
                self.get_logger().info(f"raw='{raw_output}' → STOP")
                self._save_debug_frame(curr, "stop", raw_output, goal)
                out = String(); out.data = "stop"
                self.pub_action.publish(out)
                return

            with self._lock:
                self._last_decision_frame = curr
                self._queue = [cmd] * (n_total - 1)   # 1 eseguita ora, (n-1) accodate

            self.get_logger().info(
                f"raw='{raw_output}' → {cmd} ×{n_total}  "
                f"(accodate:{n_total - 1}, hist:{len(self._frame_history)})")
            self._save_debug_frame(curr, cmd, raw_output, goal)
            out = String(); out.data = cmd
            self.pub_action.publish(out)
            # resta in volo fino al primitive_done

        except Exception as exc:
            self.get_logger().error(f"Drive error: {exc}")
            with self._lock:
                self._cycle_active = False

# =============================================================================
# HELPER METHODS
# =============================================================================
    def _wait_fresh_frame(self, after_seq, after_mono, settle_sec, timeout_sec, poll_sec=0.02):
        """Primo frame RICEVUTO dopo after_seq e dopo (after_mono + settle_sec).
        Gate locale al PC inferenza: solo contatore frame + time.monotonic, mai
        header.stamp → robusto al setup dual-machine ZED(robot) ↔ PC(inferenza).
        Con depth=1/KEEP_LAST, _last_image_msg è sempre il più recente, quindi a
        settle scaduto è già post-settling. Ritorna (msg, stale)."""
        deadline     = time.monotonic() + float(timeout_sec)
        settle_until = after_mono + float(settle_sec)
        while rclpy.ok():
            with self._lock:
                msg = self._last_image_msg
                seq = self._frame_seq
            if msg is not None and seq > after_seq and time.monotonic() >= settle_until:
                return msg, False
            if time.monotonic() >= deadline:
                return msg, True
            time.sleep(poll_sec)
        return None, True

    @staticmethod # replica fedele di sample_and_pad_images (repo ufficiale)
    def _sample_history(history, num_frames, pad_h=512, pad_w=512):
        """Replica fedele di sample_and_pad_images (repo ufficiale).
        history: frame RGB numpy oldest→newest, corrente = ultimo.
        Pad in testa con frame NERI se la storia è più corta di num_frames,
        poi campiona num_frames-1 indici (endpoint=False, int) + frame corrente."""
        frames = list(history)
        while len(frames) < num_frames:
            frames.insert(0, np.zeros((pad_h, pad_w, 3), dtype=np.uint8))
        latest = frames[-1]
        idxs = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
        return [frames[i] for i in idxs] + [latest]

    @staticmethod
    def _expand_primitives(action, value):
        """(action, value) → (cmd_primitiva, n_totale_primitive), come da repo.
        forward: step da 25 cm; turn: step da 15°."""
        if action == "forward":
            n = max(1, int(value) // 25)
            return "forward 25 cm", n
        if action == "turn_left":
            n = max(1, int(value) // 15)
            return "turn_left 15 deg", n
        if action == "turn_right":
            n = max(1, int(value) // 15)
            return "turn_right 15 deg", n
        return "stop", 0   # stop
    

# =============================================================================
# Debug
    def _save_debug_frame(self, frame_rgb: np.ndarray, action: str, raw_output: str, goal: str):
        """Save the inference frame with action overlay for debugging."""
        try:
            debug_dir = Path("/home/ros_ws/debug_frames")
            debug_dir.mkdir(exist_ok=True)

            # Converti RGB → BGR per OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Overlay testo
            timestamp = time.strftime("%H:%M:%S")
            texts = [
                f"ACTION: {action}",
                f"GOAL: {goal[:50]}",
                f"RAW: {raw_output[:60]}",
                f"TIME: {timestamp}",
                f"HISTORY: {len(self._frame_history)} frames",
            ]

            # Sfondo semitrasparente per leggibilità
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 30 + len(texts) * 28), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

            # Scrivi testo
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)  # action in verde
                cv2.putText(frame_bgr, text, (10, 25 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Salva con timestamp + action come nome file
            filename = debug_dir / f"{int(time.time()*1000)}_{action}.jpg"
            cv2.imwrite(str(filename), frame_bgr)

        except Exception as e:
            self.get_logger().warn(f"Debug frame save error: {e}", throttle_duration_sec=5.0)

# =============================================================================
# Entry point
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = NaViLANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
