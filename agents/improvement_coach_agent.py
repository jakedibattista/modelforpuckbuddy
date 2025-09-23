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

    # Build what went well
    ww: List[str] = []
    if hip_good_times:
        ww.append(
            f"Strong hip drive on {len(hip_good_times)} shot(s) (e.g., at {hip_good_times[0]:.2f}s)."
        )
    if knee_good_times:
        best = min(knee_good_times, key=lambda x: x[1])
        ww.append(f"Good front knee bend on some reps (lowest ~{best[1]:.0f}°).")
    if steady_times:
        ww.append("Smooth pull-back on several reps — nice setup tempo.")
    # Focus on core body positions only
    if excellent_head_times:
        ww.append("Excellent head position — eyes forward and level toward target!")
    if excellent_upper_times:
        ww.append("Great upper body form — shoulders level, arms extended nicely.")
    if good_back_leg_times:
        ww.append("Good back leg extension — driving power through properly.")
    if not ww:
        ww.append("Effort and repetition — good work building consistency.")

    # Build what to work on
    wt: List[str] = []
    if knee_shallow_degs:
        avg_deg = sum(knee_shallow_degs) / len(knee_shallow_degs)
        wt.append(f"Bend your FRONT knee deeper — aim ~110° front knee bend (most were ~{avg_deg:.0f}°).")
    if hip_weak_times:
        wt.append(
            f"Drive hips forward through the puck (e.g., at {hip_weak_times[0]:.2f}s)."
        )
    if jerky_times:
        wt.append("Steady the pull-back tempo — avoid rushed, choppy setup.")
    # Removed follow-through analysis - focusing on core body positions
    if bent_back_leg_times:
        worst = min(bent_back_leg_times, key=lambda x: x[1])
        wt.append(f"Straighten that back leg — extend to ~170° (currently ~{worst[1]:.0f}°).")
    if poor_head_times:
        wt.append("Keep head up and forward — eyes on target throughout the shot.")
    if poor_upper_times:
        wt.append("Square up your shoulders and extend both arms toward target.")
    if not wt:
        wt.append("Keep building strength and rhythm — maintain good habits.")

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
        "You are a supportive youth hockey coach. Rewrite the DRAFT into the SAME two sections, keeping meanings intact.\n"
        "Accuracy rules:\n"
        "- Front knee bend: LOWER degrees = deeper. Ideal ≈ 90-110°. ≤110° = good; ≥140° = shallow (not good).\n"
        "- Hip drive: 0..1; ≥0.3 = good drive. Wrist steadiness: smooth/mixed/jerky labels only.\n"
        "- Enhanced form analysis: Head position (eyes forward/level), upper body square (shoulders/arms), back leg extension (should be ~170°).\n"
        "- Back leg: 160-180° = good extension; <150° = too bent, needs straightening.\n"
        "- Form scores 0-1: ≥0.8 = excellent; 0.6-0.79 = good; <0.6 = needs work.\n"
        "Output constraints: Keep exactly two sections titled 'What went well:' and 'What to work on:' with 2–3 concise bullets each. No extra text."
    )

    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": draft}]}],
        config={
            "system_instruction": system,
            "temperature": 0.1,
            "max_output_tokens": 220,
        },
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


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


