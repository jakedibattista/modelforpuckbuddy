#!/usr/bin/env python3
"""Seth Shooting Agent (Gemini Flash Lite)

Generates TWO sections only from a drill JSON:
- What went well: 2 to 3 short bullets
- What to work on: 2 to 3 short bullets

Usage:
  python seth_shooting_agent.py results/drill/foo_drill_feedback.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.config import load_env
from utils.io import load_json_file
load_env()

try:
    from google import genai  # type: ignore
except Exception as e:
    raise RuntimeError("google-genai client not installed. pip install google-genai") from e


 


def format_timestamp(time_sec: float) -> str:
    """Format time in seconds to MM:SS format."""
    try:
        total_seconds = int(round(float(time_sec)))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "N/A"


def _summarize_metrics(raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    shots = raw.get("shots", []) or []
    times = [s.get("shot_time_sec", None) for s in shots]
    # Counters for traditional metrics
    hip_good_times: List[float] = []
    hip_weak_times: List[float] = []
    knee_good_times: List[Tuple[float, float]] = []  # (time, deg)
    knee_shallow_degs: List[float] = []
    steady_times: List[float] = []
    jerky_times: List[float] = []
    
    # Enhanced form analysis counters
    excellent_head_times: List[float] = []
    poor_head_times: List[float] = []
    excellent_upper_times: List[float] = []
    poor_upper_times: List[float] = []
    good_back_leg_times: List[float] = []
    bent_back_leg_times: List[Tuple[float, float]] = []  # (time, deg)

    for s in shots:
        t = float(s.get("shot_time_sec", 0.0))
        # Use front knee from enhanced analysis instead of general knee
        lower_body = s.get("lower_body_triangle", {}) or {}
        front_knee_deg = lower_body.get("front_knee_bend_deg")
        front_knee_deg = float(front_knee_deg) if front_knee_deg is not None else 180.0
        # Keep legacy for comparison but prefer front knee
        knee_deg = float(s.get("knee_bend_min_deg", 999.0))
        knee_score = float(s.get("knee_bend_score", 0.0))
        hip = float(s.get("hip_drive", 0.0))
        hip_good = bool(s.get("hip_drive_good", False))
        control = float(s.get("control_smoothness", 0.0))
        ctrl_label = "smooth" if control >= 0.6 else ("jerky" if control <= 0.3 else "mixed")
        # Focus on core body positions only - removed follow-through
        
        # Enhanced form analysis - handle new 0-100 scale data format
        head_pos = s.get("head_position", {}) or {}
        upper_body = s.get("upper_body_square", {}) or {}
        lower_body = s.get("lower_body_triangle", {}) or {}
        
        # New hip drive analysis (0-100 scale with categories)
        hip_analysis = s.get("hip_drive_analysis", {})
        if hip_analysis and hip_analysis.get("hip_drive_score") is not None:
            hip_score = hip_analysis["hip_drive_score"]
            hip_category = hip_analysis.get("hip_drive_category", "unknown")
            # Convert 0-100 scale to legacy 0-1 scale for compatibility
            hip = hip_score / 100.0
            hip_good = hip_score >= 60.0  # Good threshold for new scale
        
        # New wrist control analysis
        wrist_control = s.get("wrist_control", {})
        if wrist_control and wrist_control.get("setup_control_score") is not None:
            wrist_score = wrist_control["setup_control_score"]
            wrist_category = wrist_control.get("setup_control_category", "unknown")
            # Convert 0-100 scale to legacy 0-1 scale for compatibility
            control = wrist_score / 100.0
        
        # Analyze head position (new 0-100 scale)
        head_metrics = [
            head_pos.get("head_up_score"),
            head_pos.get("eyes_forward_score")
        ]
        valid_head = [m for m in head_metrics if m is not None and m > 0.0]
        avg_head = float(sum(valid_head)) / len(valid_head) / 100.0 if valid_head else 0.0  # Convert to 0-1 scale
        
        # Analyze upper body square (average of available metrics)  
        upper_metrics = [
            upper_body.get("shoulder_level"),
            upper_body.get("arm_extension"),
            upper_body.get("target_alignment")
        ]
        valid_upper = [m for m in upper_metrics if m is not None]
        avg_upper = float(sum(valid_upper)) / len(valid_upper) if valid_upper else 0.0
        
        # Back leg extension analysis  
        back_leg_deg = lower_body.get("back_leg_extension_deg")
        # front_knee_deg already handled above

        if hip_good:
            hip_good_times.append(t)
        else:
            if hip < 0.3:
                hip_weak_times.append(t)

        # Use front knee for more specific feedback (lower is better for front knee)
        if front_knee_deg <= 110.0:  # Good front knee bend
            knee_good_times.append((t, front_knee_deg))
        elif front_knee_deg >= 140.0:  # Shallow front knee bend
            knee_shallow_degs.append(front_knee_deg)

        if ctrl_label == "smooth":
            steady_times.append(t)
        elif ctrl_label == "jerky":
            jerky_times.append(t)

        # Focus on core body positions at point of release
        
        # Enhanced form analysis
        if avg_head >= 0.8:
            excellent_head_times.append(t)
        elif avg_head < 0.6 and avg_head > 0.0:
            poor_head_times.append(t)
            
        if avg_upper >= 0.8:
            excellent_upper_times.append(t)
        elif avg_upper < 0.6 and avg_upper > 0.0:
            poor_upper_times.append(t)
            
        if back_leg_deg is not None:
            if back_leg_deg >= 160.0:
                good_back_leg_times.append(t)
            elif back_leg_deg < 150.0:
                bent_back_leg_times.append((t, back_leg_deg))

    # Build what went well with specific shot references
    ww: List[str] = []
    if hip_good_times:
        if len(hip_good_times) >= 2:
            times_str = f"at {format_timestamp(hip_good_times[0])} and {format_timestamp(hip_good_times[1])}"
        else:
            times_str = f"at {format_timestamp(hip_good_times[0])}"
        ww.append(f"Nice hip drive {times_str} - you're really driving through the puck")
    
    if knee_good_times:
        best = min(knee_good_times, key=lambda x: x[1])
        ww.append(f"Great knee bend at {format_timestamp(best[0])} ({best[1]:.0f}°) - getting down low like the pros")
    
    if steady_times:
        if len(steady_times) >= 2:
            times_str = f"at {format_timestamp(steady_times[0])} and {format_timestamp(steady_times[1])}"
        else:
            times_str = f"at {format_timestamp(steady_times[0])}"
        ww.append(f"Really smooth setup {times_str} - tempo looked solid")
    
    if excellent_head_times:
        time_ref = format_timestamp(excellent_head_times[0])
        ww.append(f"Head position was money at {time_ref} - eyes locked on target")
    
    if excellent_upper_times:
        time_ref = format_timestamp(excellent_upper_times[0])
        ww.append(f"Upper body looked dialed in at {time_ref} - nice square shoulders")
    
    if good_back_leg_times:
        time_ref = format_timestamp(good_back_leg_times[0])
        ww.append(f"Back leg extension at {time_ref} was solid - driving power through")
    
    if not ww:
        ww.append("Effort was there - let's keep working on the fundamentals")

    # Build what to work on with specific shot references
    wt: List[str] = []
    if knee_shallow_degs:
        avg_deg = sum(knee_shallow_degs) / len(knee_shallow_degs)
        wt.append(f"Need to get lower on that front knee - aim for 100-110° (you're at {avg_deg:.0f}°)")
    
    if hip_weak_times:
        if len(hip_weak_times) >= 2:
            times_str = f"at {format_timestamp(hip_weak_times[0])} and {format_timestamp(hip_weak_times[1])}"
        else:
            times_str = f"at {format_timestamp(hip_weak_times[0])}"
        wt.append(f"Drive those hips harder {times_str} - really push through the puck")
    
    if jerky_times:
        time_ref = format_timestamp(jerky_times[0])
        wt.append(f"Setup looked a bit rushed at {time_ref} - take your time and be smooth")
    
    if bent_back_leg_times:
        worst = min(bent_back_leg_times, key=lambda x: x[1])
        wt.append(f"Straighten that back leg at {format_timestamp(worst[0])} - need it closer to 170° (you're at {worst[1]:.0f}°)")
    
    if poor_head_times:
        time_ref = format_timestamp(poor_head_times[0])
        wt.append(f"Head dropped a bit at {time_ref} - keep those eyes up and locked on target")
    
    if poor_upper_times:
        time_ref = format_timestamp(poor_upper_times[0])
        wt.append(f"Square up those shoulders at {time_ref} - both arms extending through")
    
    if not wt:
        wt.append("Keep grinding - consistency is key to building muscle memory")

    # Trim to 2–3 each
    return ww[:3], wt[:3]


def generate_sections(raw: Dict[str, Any], model: str = "gemini-2.5-flash-lite") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)

    # Produce rule-based draft to ensure accuracy
    ww, wt = _summarize_metrics(raw)
    draft = "**What went well:**\n" + "\n".join(f"- {b}" for b in ww) + "\n\n" + "**What to work on:**\n" + "\n".join(f"- {b}" for b in wt)

    system = (
        "You are Seth, an assistant hockey coach texting feedback after practice. Rewrite the DRAFT to sound like authentic coaching messages.\n\n"
        
        "CRITICAL: Keep ALL specific timestamps and measurements from the draft exactly as written.\n"
        "- Preserve all time references (e.g., 'at 00:08', 'at 00:08 and 00:15')\n"
        "- Keep all degree measurements (e.g., '92°', 'you're at 140°')\n"
        "- Maintain all technical details and comparisons\n\n"
        
        "COACHING VOICE - Sound like a real assistant coach:\n"
        "- Casual but knowledgeable ('that was money', 'really driving through', 'need to get lower')\n"
        "- Direct and honest without being harsh\n"
        "- Mix praise with specific areas to improve\n"
        "- Use hockey slang naturally ('dialed in', 'locked on target', 'driving power through')\n"
        "- Sound like you actually watched their shots\n\n"
        
        "TONE EXAMPLES:\n"
        "✅ 'Head position was money at 00:08 - eyes locked on target'\n"
        "✅ 'Need to get lower on that front knee - aim for 100-110° (you're at 145°)'\n"
        "✅ 'Really smooth setup at 00:15 - tempo looked solid'\n"
        "❌ 'Excellent head positioning demonstrates strong fundamentals'\n"
        "❌ 'Continue to work on improving your technique'\n\n"
        
        "Technical knowledge (unchanged from draft):\n"
        "- Front knee: Lower degrees = better (90-110° ideal)\n"
        "- Hip drive: Forward momentum through the puck\n"
        "- Back leg: Should extend 160-180° for power transfer\n\n"
        
        "Output format: Exactly two sections '**What went well:**' and '**What to work on:**' with 2-3 bullets each.\n"
        "Start directly with '**What went well:**' - no intro text."
    )

    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": draft}]}],
        config={
            "system_instruction": system,
            "temperature": 0.1,
            "max_output_tokens": 512,
        },
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    
    # Remove any unwanted introductory text that Gemini might add
    lines = text.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('what went well') or line.strip().lower().startswith('**what went well'):
            start_idx = i
            break
    
    # Return only the content starting from "What went well:"
    return '\n'.join(lines[start_idx:]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seth Shooting Agent")
    parser.add_argument("json_path", help="Path to drill feedback JSON file")
    args = parser.parse_args()

    p = Path(args.json_path)
    if not p.exists():
        raise FileNotFoundError(p)
    raw = load_json_file(p)
    text = generate_sections(raw)
    print(text)


if __name__ == "__main__":
    main()


