#!/usr/bin/env python3
"""
Hockey Shooting Drill Feedback (Pose-driven)

Pipeline:
- FFmpeg normalizes input (fps/scale/codec)
- MediaPipe Pose extracts per-frame landmarks (knees, hips, wrists)
- Pose-based phase detection (control: wrist moving back; shot: wrist burst)
- Metrics (per shot):
  - Knee bend (degrees, from shot window) and normalized score (0..1)
  - Hip drive (0..1): peak forward hip velocity during shot; good if >= 0.3
  - Control smoothness (0..1): lower wrist-speed std in control window is better

Output: concise JSON with phase times and metrics.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
 

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_pose = mp.solutions.pose
except Exception:
    MP_AVAILABLE = False
    mp_pose = None

def run_ffmpeg_normalize(input_path: str, output_path: str, fps: int = 30, width: int = 960) -> None:
    """Normalize video: h264, aac, target fps, scale by width (keep aspect)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"fps={fps},scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("⚠️ FFmpeg normalization failed, continuing with original file.")


 


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=np.float64)
    v2 = np.array([cx - bx, cy - by], dtype=np.float64)
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def pose_time_series(video_path: str, stride: int = 2) -> Tuple[List[float], Dict[str, List[Optional[Tuple[float, float, float]]]], float, int]:
    """Extract per-frame landmarks using MediaPipe Pose.

    Returns times (sec), landmarks dict with pixel coords+visibility, fps, total_frames.
    """
    if not MP_AVAILABLE:
        return [], {}, 30.0, 0
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    times: List[float] = []
    series: Dict[str, List[Optional[Tuple[float, float, float]]]] = {
        'LEFT_SHOULDER': [], 'RIGHT_SHOULDER': [],
        'LEFT_HIP': [], 'RIGHT_HIP': [], 'LEFT_KNEE': [], 'RIGHT_KNEE': [],
        'LEFT_ANKLE': [], 'RIGHT_ANKLE': [], 'LEFT_WRIST': [], 'RIGHT_WRIST': [],
        # Enhanced landmarks for form analysis
        'NOSE': [], 'LEFT_EYE': [], 'RIGHT_EYE': [],
        'LEFT_ELBOW': [], 'RIGHT_ELBOW': [],
        'LEFT_HEEL': [], 'RIGHT_HEEL': [],
        'LEFT_FOOT_INDEX': [], 'RIGHT_FOOT_INDEX': []
    }
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            ok, frame_bgr = cap.retrieve()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            res = pose.process(frame_rgb)
            def add(name: str, val: Optional[Tuple[float, float, float]]):
                series[name].append(val)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                def get_lm(i: int) -> Tuple[float, float, float]:
                    return (float(lm[i].x) * w, float(lm[i].y) * h, float(lm[i].visibility))
                # Core landmarks (existing)
                add('LEFT_SHOULDER', get_lm(mp_pose.PoseLandmark.LEFT_SHOULDER.value))
                add('RIGHT_SHOULDER', get_lm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value))
                add('LEFT_HIP', get_lm(mp_pose.PoseLandmark.LEFT_HIP.value))
                add('RIGHT_HIP', get_lm(mp_pose.PoseLandmark.RIGHT_HIP.value))
                add('LEFT_KNEE', get_lm(mp_pose.PoseLandmark.LEFT_KNEE.value))
                add('RIGHT_KNEE', get_lm(mp_pose.PoseLandmark.RIGHT_KNEE.value))
                add('LEFT_ANKLE', get_lm(mp_pose.PoseLandmark.LEFT_ANKLE.value))
                add('RIGHT_ANKLE', get_lm(mp_pose.PoseLandmark.RIGHT_ANKLE.value))
                add('LEFT_WRIST', get_lm(mp_pose.PoseLandmark.LEFT_WRIST.value))
                add('RIGHT_WRIST', get_lm(mp_pose.PoseLandmark.RIGHT_WRIST.value))
                # Enhanced landmarks for form analysis
                add('NOSE', get_lm(mp_pose.PoseLandmark.NOSE.value))
                add('LEFT_EYE', get_lm(mp_pose.PoseLandmark.LEFT_EYE.value))
                add('RIGHT_EYE', get_lm(mp_pose.PoseLandmark.RIGHT_EYE.value))
                add('LEFT_ELBOW', get_lm(mp_pose.PoseLandmark.LEFT_ELBOW.value))
                add('RIGHT_ELBOW', get_lm(mp_pose.PoseLandmark.RIGHT_ELBOW.value))
                add('LEFT_HEEL', get_lm(mp_pose.PoseLandmark.LEFT_HEEL.value))
                add('RIGHT_HEEL', get_lm(mp_pose.PoseLandmark.RIGHT_HEEL.value))
                add('LEFT_FOOT_INDEX', get_lm(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value))
                add('RIGHT_FOOT_INDEX', get_lm(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value))
            else:
                for k in series.keys():
                    add(k, None)
            times.append(idx / fps)
            idx += 1
    cap.release()
    return times, series, fps, total

def _choose_leg(series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> str:
    lv = sum(1 for v in series['LEFT_KNEE'] if v and v[2] > 0.3)
    rv = sum(1 for v in series['RIGHT_KNEE'] if v and v[2] > 0.3)
    return 'LEFT' if lv >= rv else 'RIGHT'

def knee_angle_series(series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> List[Optional[float]]:
    side = _choose_leg(series)
    hip_key, knee_key, ankle_key = f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE'
    out: List[Optional[float]] = []
    for hip, knee, ankle in zip(series[hip_key], series[knee_key], series[ankle_key]):
        if not hip or not knee or not ankle or min(hip[2], knee[2], ankle[2]) <= 0.3:
            out.append(None)
            continue
        ang = _angle_deg((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))
        out.append(ang)
    return out

def wrist_speed_series(series: Dict[str, List[Optional[Tuple[float, float, float]]]], fps: float, stride: int,
                       ema_alpha: float = 0.25, max_ffill: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def comp_speed(seq_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = series.get(seq_key, [])
        n = len(pts)
        # Build validity by visibility
        valid = np.array([(p is not None and p[2] > 0.5) for p in pts], dtype=bool)
        xs: List[Optional[float]] = [p[0] if (p and p[2] > 0.5) else None for p in pts]
        ys: List[Optional[float]] = [p[1] if (p and p[2] > 0.5) else None for p in pts]
        # EMA smoothing with brief forward-fill
        sx: Optional[float] = None
        sy: Optional[float] = None
        gap = 0
        fx = [None] * n
        fy = [None] * n
        for i in range(n):
            if xs[i] is not None and ys[i] is not None:
                if sx is None:
                    sx, sy = float(xs[i]), float(ys[i])
                else:
                    sx = float(ema_alpha * float(xs[i]) + (1.0 - ema_alpha) * sx)
                    sy = float(ema_alpha * float(ys[i]) + (1.0 - ema_alpha) * sy)
                fx[i], fy[i] = sx, sy
                gap = 0
            else:
                gap += 1
                if sx is not None and gap <= max_ffill:
                    fx[i], fy[i] = sx, sy
                else:
                    fx[i], fy[i] = None, None
        # Velocity and speed
        vx = np.zeros(n, dtype=np.float64)
        sp = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if fx[i] is None or fx[i-1] is None or fy[i] is None or fy[i-1] is None:
                continue
            dx = (fx[i] - fx[i-1]) * (fps / max(1, stride))
            dy = (fy[i] - fy[i-1]) * (fps / max(1, stride))
            vx[i] = dx
            sp[i] = float(np.hypot(dx, dy))
        return vx, sp, valid
    lvx, lsp, lvalid = comp_speed('LEFT_WRIST')
    rvx, rsp, rvalid = comp_speed('RIGHT_WRIST')
    best_sp = np.maximum(lsp, rsp)
    is_left_best = lsp >= rsp
    best_vx = np.where(is_left_best, lvx, rvx)
    return best_vx, best_sp, is_left_best.astype(np.bool_), lvalid, rvalid

 


# -------------------------------
# Enhanced Form Analysis Functions
# -------------------------------

def _detect_shooting_side(pose_series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> str:
    """Detect if player is a left or right handed shooter based on wrist movement patterns."""
    left_wrist = pose_series.get('LEFT_WRIST', [])
    right_wrist = pose_series.get('RIGHT_WRIST', [])
    
    # Count valid frames for each wrist
    left_valid = sum(1 for w in left_wrist if w and w[2] > 0.5)
    right_valid = sum(1 for w in right_wrist if w and w[2] > 0.5)
    
    # Calculate average movement range for each wrist
    left_movement = 0.0
    right_movement = 0.0
    
    if left_valid > 5:
        valid_left = [w for w in left_wrist if w and w[2] > 0.5]
        left_xs = [w[0] for w in valid_left]
        left_movement = max(left_xs) - min(left_xs) if left_xs else 0.0
    
    if right_valid > 5:
        valid_right = [w for w in right_wrist if w and w[2] > 0.5]
        right_xs = [w[0] for w in valid_right]
        right_movement = max(right_xs) - min(right_xs) if right_xs else 0.0
    
    # The top hand (more movement) indicates shooting side
    return 'RIGHT' if right_movement > left_movement else 'LEFT'

def _determine_front_back_legs(shooting_side: str, pose_series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> Tuple[str, str]:
    """Determine front and back legs based on shooting side and foot positioning."""
    # For hockey shooting stance:
    # Right-handed shooter: left leg front, right leg back
    # Left-handed shooter: right leg front, left leg back
    if shooting_side == 'RIGHT':
        return 'LEFT', 'RIGHT'  # front_leg, back_leg
    else:
        return 'RIGHT', 'LEFT'  # front_leg, back_leg

def _calculate_head_position_metrics(pose_series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> List[Dict]:
    """Calculate head position and eye level metrics for each frame."""
    nose = pose_series.get('NOSE', [])
    left_eye = pose_series.get('LEFT_EYE', [])
    right_eye = pose_series.get('RIGHT_EYE', [])
    left_shoulder = pose_series.get('LEFT_SHOULDER', [])
    right_shoulder = pose_series.get('RIGHT_SHOULDER', [])
    
    head_metrics = []
    
    for i in range(len(nose)):
        metrics = {
            'forward_lean': None,
            'eye_level': None,
            'head_direction': None
        }
        
        n = nose[i] if i < len(nose) else None
        le = left_eye[i] if i < len(left_eye) else None
        re = right_eye[i] if i < len(right_eye) else None
        ls = left_shoulder[i] if i < len(left_shoulder) else None
        rs = right_shoulder[i] if i < len(right_shoulder) else None
        
        if n and ls and rs and n[2] > 0.5 and ls[2] > 0.5 and rs[2] > 0.5:
            # Forward lean: nose position relative to shoulder line
            shoulder_center_x = (ls[0] + rs[0]) * 0.5
            forward_lean = (n[0] - shoulder_center_x) / abs(ls[0] - rs[0] + 1e-6)
            metrics['forward_lean'] = float(np.clip(abs(forward_lean), 0.0, 1.0))
        
        if le and re and le[2] > 0.5 and re[2] > 0.5:
            # Eye level consistency (both eyes at same height)
            eye_level_diff = abs(le[1] - re[1]) / (abs(le[1] + re[1]) * 0.5 + 1e-6)
            metrics['eye_level'] = float(np.clip(1.0 - eye_level_diff, 0.0, 1.0))
            
            # Head direction (eyes pointing toward target)
            eye_center_x = (le[0] + re[0]) * 0.5
            if ls and rs and ls[2] > 0.5 and rs[2] > 0.5:
                shoulder_center_x = (ls[0] + rs[0]) * 0.5
                head_direction = (eye_center_x - shoulder_center_x) / abs(ls[0] - rs[0] + 1e-6)
                metrics['head_direction'] = float(np.clip(abs(head_direction), 0.0, 1.0))
        
        head_metrics.append(metrics)
    
    return head_metrics

def _calculate_upper_body_square_metrics(pose_series: Dict[str, List[Optional[Tuple[float, float, float]]]]) -> List[Dict]:
    """Calculate upper body 'square' formation metrics for each frame."""
    left_shoulder = pose_series.get('LEFT_SHOULDER', [])
    right_shoulder = pose_series.get('RIGHT_SHOULDER', [])
    left_elbow = pose_series.get('LEFT_ELBOW', [])
    right_elbow = pose_series.get('RIGHT_ELBOW', [])
    left_wrist = pose_series.get('LEFT_WRIST', [])
    right_wrist = pose_series.get('RIGHT_WRIST', [])
    
    upper_body_metrics = []
    
    for i in range(max(len(left_shoulder), len(right_shoulder))):
        metrics = {
            'shoulder_level': None,
            'arm_extension': None,
            'target_alignment': None
        }
        
        ls = left_shoulder[i] if i < len(left_shoulder) else None
        rs = right_shoulder[i] if i < len(right_shoulder) else None
        le = left_elbow[i] if i < len(left_elbow) else None
        re = right_elbow[i] if i < len(right_elbow) else None
        lw = left_wrist[i] if i < len(left_wrist) else None
        rw = right_wrist[i] if i < len(right_wrist) else None
        
        # Shoulder level (both shoulders at same height)
        if ls and rs and ls[2] > 0.5 and rs[2] > 0.5:
            shoulder_diff = abs(ls[1] - rs[1]) / (abs(ls[1] + rs[1]) * 0.5 + 1e-6)
            metrics['shoulder_level'] = float(np.clip(1.0 - shoulder_diff, 0.0, 1.0))
        
        # Arm extension (elbows extended)
        arm_extensions = []
        for shoulder, elbow, wrist in [(ls, le, lw), (rs, re, rw)]:
            if shoulder and elbow and wrist and min(shoulder[2], elbow[2], wrist[2]) > 0.5:
                # Calculate arm extension angle
                arm_angle = _angle_deg((shoulder[0], shoulder[1]), (elbow[0], elbow[1]), (wrist[0], wrist[1]))
                # Good extension is close to 180 degrees
                extension_score = 1.0 - abs(180.0 - arm_angle) / 180.0
                arm_extensions.append(max(0.0, extension_score))
        
        if arm_extensions:
            metrics['arm_extension'] = float(np.mean(arm_extensions))
        
        # Target alignment (arms pointing same direction)
        if ls and rs and lw and rw and min(ls[2], rs[2], lw[2], rw[2]) > 0.5:
            # Calculate arm directions
            left_arm_vector = (lw[0] - ls[0], lw[1] - ls[1])
            right_arm_vector = (rw[0] - rs[0], rw[1] - rs[1])
            
            # Normalize vectors
            left_norm = np.linalg.norm(left_arm_vector) + 1e-6
            right_norm = np.linalg.norm(right_arm_vector) + 1e-6
            left_unit = (left_arm_vector[0] / left_norm, left_arm_vector[1] / left_norm)
            right_unit = (right_arm_vector[0] / right_norm, right_arm_vector[1] / right_norm)
            
            # Calculate alignment (dot product)
            alignment = np.dot(left_unit, right_unit)
            metrics['target_alignment'] = float(np.clip((alignment + 1.0) * 0.5, 0.0, 1.0))
        
        upper_body_metrics.append(metrics)
    
    return upper_body_metrics

def _calculate_lower_body_triangle_metrics(pose_series: Dict[str, List[Optional[Tuple[float, float, float]]]], 
                                          front_leg: str, back_leg: str) -> List[Dict]:
    """Calculate lower body 'triangle' formation metrics for each frame."""
    front_hip = pose_series.get(f'{front_leg}_HIP', [])
    front_knee = pose_series.get(f'{front_leg}_KNEE', [])
    front_ankle = pose_series.get(f'{front_leg}_ANKLE', [])
    front_heel = pose_series.get(f'{front_leg}_HEEL', [])
    
    back_hip = pose_series.get(f'{back_leg}_HIP', [])
    back_knee = pose_series.get(f'{back_leg}_KNEE', [])
    back_ankle = pose_series.get(f'{back_leg}_ANKLE', [])
    back_heel = pose_series.get(f'{back_leg}_HEEL', [])
    
    left_foot = pose_series.get('LEFT_FOOT_INDEX', [])
    right_foot = pose_series.get('RIGHT_FOOT_INDEX', [])
    
    lower_body_metrics = []
    
    for i in range(max(len(front_hip), len(back_hip))):
        metrics = {
            'front_knee_bend': None,
            'back_leg_extension': None,
            'stance_width': None
        }
        
        fh = front_hip[i] if i < len(front_hip) else None
        fk = front_knee[i] if i < len(front_knee) else None
        fa = front_ankle[i] if i < len(front_ankle) else None
        
        bh = back_hip[i] if i < len(back_hip) else None
        bk = back_knee[i] if i < len(back_knee) else None
        ba = back_ankle[i] if i < len(back_ankle) else None
        
        lf = left_foot[i] if i < len(left_foot) else None
        rf = right_foot[i] if i < len(right_foot) else None
        
        # Front knee bend
        if fh and fk and fa and min(fh[2], fk[2], fa[2]) > 0.5:
            front_angle = _angle_deg((fh[0], fh[1]), (fk[0], fk[1]), (fa[0], fa[1]))
            metrics['front_knee_bend'] = front_angle
        
        # Back leg extension
        if bh and bk and ba and min(bh[2], bk[2], ba[2]) > 0.5:
            back_angle = _angle_deg((bh[0], bh[1]), (bk[0], bk[1]), (ba[0], ba[1]))
            metrics['back_leg_extension'] = back_angle
        
        # Stance width (distance between feet relative to hip width)
        if lf and rf and lf[2] > 0.5 and rf[2] > 0.5:
            foot_distance = np.hypot(lf[0] - rf[0], lf[1] - rf[1])
            if fh and bh and fh[2] > 0.5 and bh[2] > 0.5:
                hip_width = np.hypot(fh[0] - bh[0], fh[1] - bh[1])
                stance_ratio = foot_distance / (hip_width + 1e-6)
                metrics['stance_width'] = float(np.clip(stance_ratio, 0.0, 3.0))
        
        lower_body_metrics.append(metrics)
    
    return lower_body_metrics


# -------------------------------
# Multi-shot helpers
# -------------------------------
def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    m = np.nanmean(a)
    s = np.nanstd(a) + 1e-6
    return (a - m) / s

def find_shot_peaks(wrist_speed: np.ndarray, fps: float, stride: int,
                    min_sep_s: float = 2.0, thresh_z: float = 1.5,
                    min_height_frac: float = 0.2) -> List[int]:
    if wrist_speed.size == 0:
        return []
    z = _zscore(wrist_speed)
    min_sep = int((fps / max(1, stride)) * min_sep_s)
    peaks: List[int] = []
    last = -min_sep
    max_speed = float(np.max(wrist_speed)) if wrist_speed.size else 0.0
    min_height = max_speed * float(max(0.0, min_height_frac))
    for i in range(1, len(z) - 1):
        if z[i] > thresh_z and z[i] > z[i-1] and z[i] > z[i+1] and wrist_speed[i] >= min_height:
            if i - last >= min_sep:
                peaks.append(i)
                last = i
    # Fallback: if none found, take global max
    if not peaks:
        peaks = [int(np.argmax(wrist_speed))]
    return peaks

def window_for_peak(i: int, wrist_vx: np.ndarray, n: int, fps: float, stride: int,
                    max_back_s: float = 1.5, shot_forward_s: float = 0.25,
                    allow_gap_s: float = 0.2) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    # control window: scan back allowing brief positive-vx gaps, limited by max_back_s
    max_back = int((fps / max(1, stride)) * max_back_s)
    allow_gap = max(0, int((fps / max(1, stride)) * allow_gap_s))
    start = i
    steps = 0
    consecutive_pos = 0
    while start > 0 and steps < max_back:
        if wrist_vx[start] <= 0:
            consecutive_pos = 0
        else:
            consecutive_pos += 1
            if consecutive_pos > allow_gap:
                break
        start -= 1
        steps += 1
    control = (max(0, start), max(0, i - 1))
    # shot window: small forward window
    shot_fw = int((fps / max(1, stride)) * shot_forward_s)
    shot = (i, min(n - 1, i + max(1, shot_fw)))
    return control, shot


 


def _normalize_video(video_path: str) -> str:
    """Normalize video using FFmpeg and return the path to use for analysis.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Path to normalized video if successful, otherwise original path
    """
    # Allow worker to control temp/output directory via PB_WORK_DIR
    base_dir = os.getenv("PB_WORK_DIR", "videos/processed")
    norm_path = str(Path(base_dir) / (Path(video_path).stem + "_norm.mp4"))
    run_ffmpeg_normalize(video_path, norm_path, fps=30, width=960)
    use_path = norm_path if Path(norm_path).exists() else video_path
    return use_path


def _extract_pose_features(video_path: str, stride: int = 2) -> Tuple:
    """Extract pose landmarks and compute all derived features.
    
    Args:
        video_path: Path to the video file
        stride: Frame stride for pose extraction
        
    Returns:
        Tuple containing all computed features:
        (times, pose_series, fps, total_frames, duration_sec, knees, wrist_vx, 
         wrist_speed, dominant_valid, hip_vx, torso_height, active_wrist_y, active_shoulder_y)
    """
    # Pose time series (strided for speed)
    times, pose_series, fps, total_frames = pose_time_series(video_path, stride=stride)
    duration_sec = float(total_frames) / max(1.0, fps) if total_frames else (times[-1] if times else 0.0)

    # Knee angle series and wrist speeds
    knees = knee_angle_series(pose_series)
    wrist_vx, wrist_speed, is_left_best, lvalid, rvalid = wrist_speed_series(
        pose_series, fps=fps, stride=stride
    )

    # Dominant wrist validity per frame
    dominant_valid = np.where(is_left_best, lvalid, rvalid)

    # Build hips_x and hip_vx once (used in per-shot metrics)
    hips_x: List[Optional[float]] = []
    for lh, rh in zip(pose_series.get('LEFT_HIP', []), pose_series.get('RIGHT_HIP', [])):
        if lh and rh and lh[2] > 0.3 and rh[2] > 0.3:
            hips_x.append((lh[0] + rh[0]) * 0.5)
        else:
            hips_x.append(None)
    hip_vx = np.zeros(len(hips_x), dtype=np.float64)
    for i in range(1, len(hips_x)):
        if hips_x[i] is None or hips_x[i-1] is None:
            continue
        hip_vx[i] = (hips_x[i] - hips_x[i-1]) * (fps / max(1, stride))

    # Precompute torso height and wrist/shoulder Y with active side selection
    left_shoulder = pose_series.get('LEFT_SHOULDER', [])
    right_shoulder = pose_series.get('RIGHT_SHOULDER', [])
    left_wrist = pose_series.get('LEFT_WRIST', [])
    right_wrist = pose_series.get('RIGHT_WRIST', [])
    left_hip = pose_series.get('LEFT_HIP', [])
    right_hip = pose_series.get('RIGHT_HIP', [])

    torso_height: List[Optional[float]] = []
    active_wrist_y: List[Optional[float]] = []
    active_shoulder_y: List[Optional[float]] = []
    for i in range(len(times)):
        ls = left_shoulder[i] if i < len(left_shoulder) else None
        rs = right_shoulder[i] if i < len(right_shoulder) else None
        lh = left_hip[i] if i < len(left_hip) else None
        rh = right_hip[i] if i < len(right_hip) else None
        # Torso height: shoulder to hip vertical distance (average of sides)
        torso_val: Optional[float] = None
        vals: List[float] = []
        if ls and lh and ls[2] > 0.3 and lh[2] > 0.3:
            vals.append(abs(lh[1] - ls[1]))
        if rs and rh and rs[2] > 0.3 and rh[2] > 0.3:
            vals.append(abs(rh[1] - rs[1]))
        if vals:
            torso_val = float(np.mean(vals))
        torso_height.append(torso_val)
        # Active side wrist/shoulder y (choose based on dominant wrist speed at frame)
        use_left = bool(is_left_best[i]) if i < len(is_left_best) else True
        wr = left_wrist[i] if use_left else right_wrist[i]
        sh = ls if use_left else rs
        if wr and sh and wr[2] > 0.3 and sh[2] > 0.3:
            active_wrist_y.append(float(wr[1]))
            active_shoulder_y.append(float(sh[1]))
        else:
            active_wrist_y.append(None)
            active_shoulder_y.append(None)
    
    # Enhanced form analysis
    shooting_side = _detect_shooting_side(pose_series)
    front_leg, back_leg = _determine_front_back_legs(shooting_side, pose_series)
    head_metrics = _calculate_head_position_metrics(pose_series)
    upper_body_metrics = _calculate_upper_body_square_metrics(pose_series)
    lower_body_metrics = _calculate_lower_body_triangle_metrics(pose_series, front_leg, back_leg)
    
    return (times, pose_series, fps, total_frames, duration_sec, knees, wrist_vx, 
            wrist_speed, dominant_valid, hip_vx, torso_height, active_wrist_y, active_shoulder_y,
            shooting_side, front_leg, back_leg, head_metrics, upper_body_metrics, lower_body_metrics)


def _detect_and_analyze_shots(times: List[float], fps: float, stride: int, knees: List[Optional[float]], 
                              wrist_vx: np.ndarray, wrist_speed: np.ndarray, dominant_valid: np.ndarray,
                              hip_vx: np.ndarray, active_wrist_y: List[Optional[float]], 
                              active_shoulder_y: List[Optional[float]], torso_height: List[Optional[float]],
                              shooting_side: str, front_leg: str, back_leg: str,
                              head_metrics: List[Dict], upper_body_metrics: List[Dict], 
                              lower_body_metrics: List[Dict]) -> List[Dict]:
    """Detect shot peaks and analyze each shot's metrics.
    
    Args:
        times: Time series array
        fps: Frame rate
        stride: Frame stride used for analysis
        knees: Knee angle series
        wrist_vx: Wrist velocity in x direction
        wrist_speed: Wrist speed magnitude
        dominant_valid: Validity mask for dominant wrist
        hip_vx: Hip velocity in x direction  
        active_wrist_y: Y coordinates of active wrist
        active_shoulder_y: Y coordinates of active shoulder
        torso_height: Torso height measurements
        
    Returns:
        List of shot event dictionaries with metrics
    """
    # Multi-shot detection (stricter thresholds and min peak height)
    peak_idxs = find_shot_peaks(
        wrist_speed, fps=fps, stride=stride, min_sep_s=2.0, thresh_z=1.5, min_height_frac=0.2
    )
    shot_events = []
    for p in peak_idxs:
        control_idx, shot_idx = window_for_peak(
            p, wrist_vx, len(times), fps=fps, stride=stride,
            max_back_s=1.5, shot_forward_s=0.25, allow_gap_s=0.2
        )
        # Metrics per shot
        # Knee bend measured in the shot window
        knee_angles_shot: List[float] = [
            float(k) for k in knees[shot_idx[0]:shot_idx[1]+1]
            if (k is not None)
        ] if knees and shot_idx[1] >= shot_idx[0] else []
        knee_samples = len(knee_angles_shot)
        knee_bend_valid = bool(knee_samples >= 3)
        knee_bend_min = float(min(knee_angles_shot)) if knee_angles_shot else 0.0
        # Normalize: 1.0 at <=100°, 0.0 at >=140°, linear in between
        if knee_angles_shot:
            a = knee_bend_min
            knee_bend_score = float(np.clip((140.0 - a) / 40.0, 0.0, 1.0))
        else:
            knee_bend_score = 0.0
        # Hip drive near shot
        shot_slice = slice(shot_idx[0], min(len(times), shot_idx[1]+1))
        # Only consider forward (positive) hip vx for drive
        hip_window = hip_vx[shot_slice] if shot_slice.stop > shot_slice.start else np.array([], dtype=np.float64)
        hip_forward = hip_window[hip_window > 0.0]
        hip_drive_peak = float(np.max(hip_forward)) if hip_forward.size > 0 else 0.0
        pos_hips = hip_vx[hip_vx > 0.0]
        hip_norm_denom = float(np.nanpercentile(pos_hips, 90)) if pos_hips.size > 0 else float(np.nanpercentile(np.abs(hip_vx), 90) + 1e-6)
        hip_drive_norm = float(np.clip(hip_drive_peak / (hip_norm_denom + 1e-6), 0.0, 1.0))
        hip_drive_good = bool(hip_drive_norm >= 0.3)
        # Control smoothness with validity, min window, and direction gating
        if control_idx[1] > control_idx[0]:
            start_i, end_i = control_idx[0], control_idx[1]
            ctrl_len_sec = (end_i - start_i + 1) * (stride / max(1.0, fps))
            idxs = np.arange(start_i, end_i + 1)
            valid_mask = dominant_valid[idxs]
            valid_count = int(np.sum(valid_mask))
            # Direction gating: require majority of frames moving back (vx <= 0)
            dir_mask = wrist_vx[idxs] <= 0.0
            dir_frac = float(np.sum(dir_mask & valid_mask)) / float(max(1, valid_count)) if valid_count > 0 else 0.0
            min_ctrl_sec = 0.4
            min_ctrl_samples = 6
            control_valid = bool(
                (ctrl_len_sec >= min_ctrl_sec) and (valid_count >= min_ctrl_samples) and (dir_frac >= 0.7)
            )
            if control_valid:
                ctrl_sp = wrist_speed[idxs][valid_mask]
                # Clip-normalized scoring using 90th percentile of valid speeds across clip
                clip_valid_mask = dominant_valid.astype(bool)
                global_ref = float(np.nanpercentile(wrist_speed[clip_valid_mask], 90) + 1e-6) if np.any(clip_valid_mask) else float(np.nanpercentile(wrist_speed, 90) + 1e-6)
                ctrl_std = float(np.nanstd(ctrl_sp)) if ctrl_sp.size > 1 else 0.0
                control_smoothness = float(np.clip(1.0 - (ctrl_std / global_ref), 0.0, 1.0))
            else:
                control_smoothness = 0.0
        else:
            control_smoothness = 0.0
        # Focus on core body positions at point of release only
        # Enhanced form analysis for shot window
        shot_window_slice = slice(shot_idx[0], min(len(times), shot_idx[1]+1))
        
        # Average form metrics during shot window
        shot_head_metrics = head_metrics[shot_window_slice]
        shot_upper_metrics = upper_body_metrics[shot_window_slice]
        shot_lower_metrics = lower_body_metrics[shot_window_slice]
        
        # Calculate averages for valid frames
        def safe_average(metrics_list: List[Dict], key: str) -> Optional[float]:
            values = [m[key] for m in metrics_list if m[key] is not None]
            return float(np.mean(values)) if values else None
        
        # Head position metrics
        head_forward_lean = safe_average(shot_head_metrics, 'forward_lean')
        head_eye_level = safe_average(shot_head_metrics, 'eye_level')
        head_direction = safe_average(shot_head_metrics, 'head_direction')
        
        # Upper body square metrics
        shoulder_level = safe_average(shot_upper_metrics, 'shoulder_level')
        arm_extension = safe_average(shot_upper_metrics, 'arm_extension')
        target_alignment = safe_average(shot_upper_metrics, 'target_alignment')
        
        # Lower body triangle metrics  
        front_knee_angles = [m['front_knee_bend'] for m in shot_lower_metrics if m['front_knee_bend'] is not None]
        back_leg_angles = [m['back_leg_extension'] for m in shot_lower_metrics if m['back_leg_extension'] is not None]
        front_knee_bend_deg = float(np.mean(front_knee_angles)) if front_knee_angles else None
        back_leg_extension_deg = float(np.mean(back_leg_angles)) if back_leg_angles else None
        
        # Times
        def idx_to_time(i: int) -> float:
            return float(i * stride) / max(1.0, fps)
        shot_time = idx_to_time(p)
        control_t = (idx_to_time(control_idx[0]), idx_to_time(control_idx[1]))
        shot_t    = (idx_to_time(shot_idx[0]), idx_to_time(shot_idx[1]))
        shot_events.append({
            "shot_time_sec": round(shot_time, 3),
            # Keep legacy metrics for compatibility (agents still use)
            "knee_bend_min_deg": round(knee_bend_min, 1),
            "knee_bend_score": round(knee_bend_score, 3),
            "hip_drive": round(hip_drive_norm, 3),
            "hip_drive_good": hip_drive_good,
            "control_smoothness": round(control_smoothness, 3),
            # Enhanced form analysis (core metrics)
            "head_position": {
                "forward_lean": round(head_forward_lean, 3) if head_forward_lean is not None else None,
                "eye_level": round(head_eye_level, 3) if head_eye_level is not None else None,
                "target_facing": round(head_direction, 3) if head_direction is not None else None
            },
            "upper_body_square": {
                "shoulder_level": round(shoulder_level, 3) if shoulder_level is not None else None,
                "arm_extension": round(arm_extension, 3) if arm_extension is not None else None,
                "target_alignment": round(target_alignment, 3) if target_alignment is not None else None
            },
            "lower_body_triangle": {
                "front_knee_bend_deg": round(front_knee_bend_deg, 1) if front_knee_bend_deg is not None else None,
                "back_leg_extension_deg": round(back_leg_extension_deg, 1) if back_leg_extension_deg is not None else None
            }
        })
    return shot_events


def _format_analysis_results(video_path: str, fps: float, duration_sec: float, shot_events: List[Dict]) -> Dict:
    """Format the final analysis results dictionary.
    
    Args:
        video_path: Original video file path
        fps: Frame rate
        duration_sec: Video duration in seconds  
        shot_events: List of analyzed shot events
        
    Returns:
        Final formatted results dictionary
    """
    # Legacy top-level (first shot) for backward compatibility
    first = shot_events[0] if shot_events else None
    return {
        "video": Path(video_path).name,
        "fps": fps,
        "duration_est_sec": duration_sec,
        "shots": shot_events,
        "phases": {},  # Removed - not used by agents
        "metrics": (first and {
            "knee_bend_min_deg": first["knee_bend_min_deg"],
            "knee_bend_score": first.get("knee_bend_score", 0.0),
            "hip_drive": first["hip_drive"],
            "hip_drive_good": first.get("hip_drive_good", False),
            "control_smoothness": first["control_smoothness"],
        }) or {},
    }


def analyze_drill(video_path: str) -> Dict:
    """Analyze hockey drill video and return metrics for detected shots.
    
    This is the main entry point that orchestrates the entire analysis pipeline:
    1. Video normalization via FFmpeg
    2. Pose feature extraction via MediaPipe
    3. Shot detection and metric computation
    4. Result formatting for JSON output
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Dictionary containing analysis results with shot metrics
    """
    # Video normalization
    use_path = _normalize_video(video_path)

    # Extract pose features and compute derived data
    stride = 2
    (times, pose_series, fps, total_frames, duration_sec, knees, wrist_vx, 
     wrist_speed, dominant_valid, hip_vx, torso_height, active_wrist_y, active_shoulder_y,
     shooting_side, front_leg, back_leg, head_metrics, upper_body_metrics, lower_body_metrics) = _extract_pose_features(use_path, stride)

    # Detect and analyze shots
    shot_events = _detect_and_analyze_shots(
        times, fps, stride, knees, wrist_vx, wrist_speed, dominant_valid, 
        hip_vx, active_wrist_y, active_shoulder_y, torso_height,
        shooting_side, front_leg, back_leg, head_metrics, upper_body_metrics, lower_body_metrics
    )

    # Format final results
    return _format_analysis_results(video_path, fps, duration_sec, shot_events)


def main():
    print("🏒 Drill Feedback (MediaPipe Pose + OpenCV + FFmpeg)")
    # Allow passing a video path via CLI arg; default to sample path
    in_path = sys.argv[1] if len(sys.argv) > 1 else "videos/input/hockeyshoot (1).mov"
    if not Path(in_path).exists():
        print(f"❌ Not found: {in_path}")
        sys.exit(1)
    t0 = time.time()
    result = analyze_drill(in_path)
    out_dir = Path("results/drill")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(in_path).stem}_drill_feedback.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    dt = time.time() - t0
    print(f"✅ Feedback saved: {out_file} ({dt:.1f}s)")


if __name__ == "__main__":
    main()



