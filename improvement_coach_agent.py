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
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from google import genai  # type: ignore
except Exception as e:
    raise RuntimeError("google-genai client not installed. pip install google-genai") from e


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r") as f:
        return json.load(f)


def _summarize_metrics(raw: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    shots = raw.get("shots", []) or []
    times = [s.get("shot_time_sec", None) for s in shots]
    # Counters
    hip_good_times: List[float] = []
    hip_weak_times: List[float] = []
    knee_good_times: List[Tuple[float, float]] = []  # (time, deg)
    knee_shallow_degs: List[float] = []
    steady_times: List[float] = []
    jerky_times: List[float] = []
    follow_yes_times: List[float] = []
    follow_no_times: List[float] = []
    lift_but_weak_times: List[float] = []

    for s in shots:
        t = float(s.get("shot_time_sec", 0.0))
        knee_deg = float(s.get("knee_bend_min_deg", 999.0))
        knee_score = float(s.get("knee_bend_score", 0.0))
        hip = float(s.get("hip_drive", 0.0))
        hip_good = bool(s.get("hip_drive_good", False))
        control = float(s.get("control_smoothness", 0.0))
        ctrl_label = "smooth" if control >= 0.6 else ("jerky" if control <= 0.3 else "mixed")
        stick = s.get("stick_lift", {}) or {}
        follow = bool(stick.get("present", False))

        if hip_good:
            hip_good_times.append(t)
        else:
            if hip < 0.3:
                hip_weak_times.append(t)

        if knee_deg <= 110.0:
            knee_good_times.append((t, knee_deg))
        elif knee_deg >= 140.0:
            knee_shallow_degs.append(knee_deg)

        if ctrl_label == "smooth":
            steady_times.append(t)
        elif ctrl_label == "jerky":
            jerky_times.append(t)

        if follow:
            follow_yes_times.append(t)
            if not hip_good:
                lift_but_weak_times.append(t)
        else:
            follow_no_times.append(t)

    # Build what went well
    ww: List[str] = []
    if hip_good_times:
        ww.append(
            f"Strong hip drive on {len(hip_good_times)} shot(s) (e.g., at {hip_good_times[0]:.2f}s)."
        )
    if knee_good_times:
        best = min(knee_good_times, key=lambda x: x[1])
        ww.append(f"Good knee bend on some reps (lowest ~{best[1]:.0f}°).")
    if steady_times:
        ww.append("Smooth pull-back on several reps — nice setup tempo.")
    if follow_yes_times:
        ww.append("Nice follow-through on some shots — finished high toward target.")
    if not ww:
        ww.append("Effort and repetition — good work building consistency.")

    # Build what to work on
    wt: List[str] = []
    if knee_shallow_degs:
        avg_deg = sum(knee_shallow_degs) / len(knee_shallow_degs)
        wt.append(f"Get lower before release — aim ~110° knee bend (most were ~{avg_deg:.0f}°).")
    if hip_weak_times:
        wt.append(
            f"Drive hips forward through the puck (e.g., at {hip_weak_times[0]:.2f}s)."
        )
    if jerky_times:
        wt.append("Steady the pull-back tempo — avoid rushed, choppy setup.")
    if follow_no_times and len(follow_no_times) > len(follow_yes_times):
        wt.append("Finish your shot — extend wrists and stick toward the target.")
    if lift_but_weak_times:
        wt.append("Don’t lift early — drive through first, then finish high.")
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
        "- Knee bend: LOWER degrees = deeper. Ideal ≈ 90°. ≤110° = good; ≥140° = shallow (not good).\n"
        "- Hip drive: 0..1; ≥0.3 = good drive. Wrist steadiness: smooth/mixed/jerky labels only. Follow-through: yes/no.\n"
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
    raw = load_json(p)
    text = generate_sections(raw)
    print(text)


if __name__ == "__main__":
    main()


