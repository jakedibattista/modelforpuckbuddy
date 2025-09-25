#!/usr/bin/env python3
"""Improvement Coach Agent (Gemini Flash Lite)

Generates TWO sections only from a drill JSON:
- What went well: 2 to 3 short bullets
- What to work on: 2 to 3 short bullets

Usage:
  python improvement_coach_agent.py results/drill/foo_drill_feedback.json
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
        front_knee_deg = float(lower_body.get("front_knee_bend_deg", 180.0))
        # Keep legacy for comparison but prefer front knee
        knee_deg = float(s.get("knee_bend_min_deg", 999.0))
        knee_score = float(s.get("knee_bend_score", 0.0))
        hip = float(s.get("hip_drive", 0.0))
        hip_good = bool(s.get("hip_drive_good", False))
        control = float(s.get("control_smoothness", 0.0))
        ctrl_label = "smooth" if control >= 0.6 else ("jerky" if control <= 0.3 else "mixed")
        # Focus on core body positions only - removed follow-through
        
        # Enhanced form analysis
        head_pos = s.get("head_position", {}) or {}
        upper_body = s.get("upper_body_square", {}) or {}
        lower_body = s.get("lower_body_triangle", {}) or {}
        
        # Analyze head position (average of available metrics)
        head_metrics = [
            head_pos.get("forward_lean"),
            head_pos.get("eye_level"), 
            head_pos.get("target_facing")
        ]
        valid_head = [m for m in head_metrics if m is not None]
        avg_head = float(sum(valid_head)) / len(valid_head) if valid_head else 0.0
        
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
        front_knee_deg = lower_body.get("front_knee_bend_deg")

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
            times_str = f"shots at {format_timestamp(hip_good_times[0])}, {format_timestamp(hip_good_times[1])}"
        else:
            times_str = f"shot at {format_timestamp(hip_good_times[0])}"
        ww.append(f"Strong hip drive on {times_str} — driving power through the puck!")
    
    if knee_good_times:
        best = min(knee_good_times, key=lambda x: x[1])
        ww.append(f"Excellent front knee bend at {format_timestamp(best[0])} ({best[1]:.0f}°) — getting low for power!")
    
    if steady_times:
        if len(steady_times) >= 2:
            times_str = f"{format_timestamp(steady_times[0])}, {format_timestamp(steady_times[1])}"
        else:
            times_str = format_timestamp(steady_times[0])
        ww.append(f"Smooth, controlled setup at {times_str} — great tempo and rhythm.")
    
    if excellent_head_times:
        time_ref = format_timestamp(excellent_head_times[0])
        ww.append(f"Excellent head position at {time_ref} — eyes up and forward toward target!")
    
    if excellent_upper_times:
        time_ref = format_timestamp(excellent_upper_times[0])
        ww.append(f"Outstanding upper body form at {time_ref} — shoulders square and arms extended.")
    
    if good_back_leg_times:
        time_ref = format_timestamp(good_back_leg_times[0])
        ww.append(f"Solid back leg extension at {time_ref} — driving power through properly.")
    
    if not ww:
        ww.append("Good effort and consistency — keep building these fundamentals.")

    # Build what to work on with specific shot references
    wt: List[str] = []
    if knee_shallow_degs:
        avg_deg = sum(knee_shallow_degs) / len(knee_shallow_degs)
        wt.append(f"Get lower on your front knee — aim for 100-110° bend (currently averaging {avg_deg:.0f}°).")
    
    if hip_weak_times:
        if len(hip_weak_times) >= 2:
            times_str = f"shots at {format_timestamp(hip_weak_times[0])}, {format_timestamp(hip_weak_times[1])}"
        else:
            times_str = f"shot at {format_timestamp(hip_weak_times[0])}"
        wt.append(f"Drive hips forward more aggressively on {times_str} — push through the puck!")
    
    if jerky_times:
        time_ref = format_timestamp(jerky_times[0])
        wt.append(f"Smooth out the setup tempo (see {time_ref}) — avoid rushed, choppy movement.")
    
    if bent_back_leg_times:
        worst = min(bent_back_leg_times, key=lambda x: x[1])
        wt.append(f"Straighten that back leg at {format_timestamp(worst[0])} — extend to 170° (currently {worst[1]:.0f}°).")
    
    if poor_head_times:
        time_ref = format_timestamp(poor_head_times[0])
        wt.append(f"Keep head up and forward at {time_ref} — eyes locked on target throughout.")
    
    if poor_upper_times:
        time_ref = format_timestamp(poor_upper_times[0])
        wt.append(f"Square up your shoulders at {time_ref} — both arms extending toward target.")
    
    if not wt:
        wt.append("Focus on consistency and repetition — keep building these strong fundamentals.")

    # Trim to 2–3 each
    return ww[:3], wt[:3]


def generate_sections(raw: Dict[str, Any], model: str = "gemini-2.5-flash-lite") -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)

    # Produce rule-based draft to ensure accuracy
    ww, wt = _summarize_metrics(raw)
    draft = "What went well:\n" + "\n".join(f"- {b}" for b in ww) + "\n\n" + "What to work on:\n" + "\n".join(f"- {b}" for b in wt)

    system = (
        "You are a supportive youth hockey coach providing SPECIFIC feedback. Rewrite the DRAFT into the SAME two sections, keeping all specific details intact.\n\n"
        
        "CRITICAL: The draft contains specific shot timestamps (MM:SS format) and exact measurements - PRESERVE ALL OF THESE.\n"
        "- Keep all time references (e.g., 'at 00:08', 'shots at 00:08, 00:15')\n"
        "- Keep all degree measurements (e.g., '92°', 'currently 140°')\n"
        "- Keep all specific technical details and comparisons\n\n"
        
        "Technical accuracy rules:\n"
        "- Front knee bend: LOWER degrees = deeper/better. Ideal ≈ 90-110°. ≤110° = excellent; ≥140° = too shallow\n"
        "- Hip drive: 0-1 scale; ≥0.3 = good drive, <0.3 = needs more aggression\n"
        "- Back leg extension: 160-180° = good; <150° = too bent and loses power\n"
        "- Head/upper body form: 0-1 scale; ≥0.8 = excellent; 0.6-0.79 = good; <0.6 = needs work\n\n"
        
        "Style guidelines:\n"
        "- Use energetic, specific coaching language\n"
        "- Reference exact timestamps and measurements from the draft\n"
        "- Avoid generic phrases like 'keep practicing' or 'work on fundamentals'\n"
        "- Focus on actionable, specific improvements with concrete targets\n"
        "- Maintain positive, encouraging tone while being precise\n\n"
        
        "Output format: Exactly two sections titled 'What went well:' and 'What to work on:' with 2-3 specific bullets each.\n\n"
        "IMPORTANT: Output ONLY these two sections. Do NOT add any introductory text, greeting, or additional commentary. Start directly with 'What went well:'"
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
        if line.strip().lower().startswith('what went well'):
            start_idx = i
            break
    
    # Return only the content starting from "What went well:"
    return '\n'.join(lines[start_idx:]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Improvement Coach Agent")
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


