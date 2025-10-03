#!/usr/bin/env python3
"""
OpenIce Agent - Intelligent Hockey Shooting Coach with Conversational AI

This agent provides conversational Q&A about hockey shooting technique,
combining technical analysis data with real-time web search capabilities.

Features:
- Persistent chat sessions with context memory
- Google Search integration for current hockey knowledge
- Player comparisons and technique analysis
- Practice recommendations and drill suggestions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        f"Google GenAI library not installed: {exc}\n"
        "Install with: pip install google-genai"
    ) from exc


class OpenIceAgent:
    """
    Conversational AI coach for hockey shooting analysis.
    
    Uses Gemini's chat capabilities with Google Search integration to provide
    intelligent, context-aware responses about hockey technique.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenIce agent with Gemini client."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        self.chat_sessions: Dict[str, Any] = {}
        
    def create_chat_session(self, analysis_data: str, user_id: str = "anonymous") -> str:
        """
        Create a new chat session with technical analysis data as context.
        
        Args:
            analysis_data: Technical shooting analysis from data_summary_agent
            user_id: Optional user identifier for the session
            
        Returns:
            session_id: Unique identifier for the chat session
        """
        session_id = str(uuid.uuid4())
        
        # Create the initial system prompt with analysis context
        system_prompt = f"""You are OpenIce, a direct hockey shooting coach who gives specific, actionable feedback.

PLAYER'S SHOOTING SESSION DATA:
{analysis_data}

RESPONSE RULES:
1. ALWAYS reference specific data points from their session (scores, timestamps, angles)
2. Keep responses under 150 words - be concise and actionable
3. Focus on ONE main improvement area per response unless asked for multiple
4. Use their exact scores/categories when giving feedback
5. Give specific drills or techniques, not general advice

COACHING STYLE:
- Direct but encouraging ("Your 72/100 head position needs work" not "head position could be better")
- Data-driven ("At 00:08 your knee was 95° - aim for 100-110° for more power")
- Actionable ("Try the wall drill: stand 6 inches from wall, practice keeping head up")
- Personal ("YOUR shot at 00:15 vs shot at 00:08")

SCORE INTERPRETATION:
- 80-100: excellent (celebrate it)
- 60-79: good (minor tweaks)
- Below 60: needs work (specific improvement)

NHL COMPARISONS:
Only mention pros when specifically relevant to their data and include what technique to copy.

AVOID:
- Generic advice ("practice makes perfect")
- Long explanations of basic concepts
- Multiple improvement areas in one response
- Vague suggestions ("work on form")

Be specific, be brief, be helpful.
"""

        try:
            # Create chat session with Google Search enabled and optimized for concise responses
            chat = self.client.chats.create(
                model='gemini-2.5-flash',
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    max_output_tokens=1000,  # Limit to ~150 words for concise responses
                    temperature=0.3,        # Lower temperature for more focused responses
                    top_p=0.8              # More focused sampling for consistency
                )
            )
            
            # Prime the chat with system context
            chat.send_message(system_prompt)
            
            # Store session info
            self.chat_sessions[session_id] = {
                'chat': chat,
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'message_count': 0,
                'analysis_data': analysis_data
            }
            
            return session_id
            
        except Exception as exc:
            raise RuntimeError(f"Failed to create chat session: {exc}") from exc
    
    def _enhance_question_with_context(self, question: str, session: Dict[str, Any]) -> str:
        """Enhance user question with context hints to encourage specific responses."""
        context_hints = [
            "Reference my specific scores and timestamps.",
            "Keep your answer under 150 words.",
            "Focus on one main improvement area.",
            "Give me a specific drill or technique to practice."
        ]
        
        enhanced_question = f"{question}\n\n(Coach: {' '.join(context_hints)})"
        return enhanced_question
    
    def ask_question(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Ask a question in an existing chat session.
        
        Args:
            session_id: Chat session identifier
            question: User's question about their shooting technique
            
        Returns:
            Dictionary containing response and metadata
        """
        if session_id not in self.chat_sessions:
            raise ValueError(f"Chat session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        chat = session['chat']
        
        try:
            # Enhance question with context hints for more focused responses
            enhanced_question = self._enhance_question_with_context(question, session)
            
            # Send the enhanced question to the chat
            response = chat.send_message(enhanced_question)
            
            # Update session metadata
            session['last_activity'] = datetime.now()
            session['message_count'] += 1
            
            # Extract search metadata if available
            search_queries = []
            sources = []
            
            if (response.candidates and 
                len(response.candidates) > 0 and 
                response.candidates[0].grounding_metadata):
                
                grounding = response.candidates[0].grounding_metadata
                if grounding.web_search_queries:
                    search_queries = grounding.web_search_queries
                if grounding.grounding_chunks:
                    sources = [chunk.web.title for chunk in grounding.grounding_chunks if hasattr(chunk, 'web')]
            
            # Process and validate response quality
            answer_text = response.text.strip()
            
            # Count words to check if response is too long
            word_count = len(answer_text.split())
            
            # If response is too long, try to get a shorter version
            if word_count > 200:
                try:
                    shorter_response = chat.send_message(
                        "That response was too long. Give me the same advice in under 150 words, focusing on just the most important point."
                    )
                    answer_text = shorter_response.text.strip()
                except:
                    # If shortening fails, use original response
                    pass
            
            return {
                'answer': answer_text,
                'search_queries': search_queries,
                'sources': sources,
                'session_id': session_id,
                'message_count': session['message_count'],
                'word_count': len(answer_text.split())
            }
            
        except Exception as exc:
            raise RuntimeError(f"Failed to process question: {exc}") from exc
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a chat session."""
        if session_id not in self.chat_sessions:
            raise ValueError(f"Chat session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        return {
            'session_id': session_id,
            'user_id': session['user_id'],
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat(),
            'message_count': session['message_count']
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old chat sessions to manage memory."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            sid for sid, session in self.chat_sessions.items()
            if session['last_activity'] < cutoff
        ]
        
        for session_id in old_sessions:
            del self.chat_sessions[session_id]
        
        return len(old_sessions)


def parse_raw_pose_analysis(raw_analysis: Dict[str, Any]) -> str:
    """
    Convert raw pose_extraction_shooting_drills.py output into readable analysis text.
    
    Args:
        raw_analysis: Raw output from analyze_drill() function
        
    Returns:
        Human-readable analysis text for OpenIce coaching
    """
    try:
        # Extract key information from raw analysis
        video_name = raw_analysis.get('video', 'Unknown video')
        duration = raw_analysis.get('duration_est_sec', 0)
        fps = raw_analysis.get('fps', 30)
        shots = raw_analysis.get('shots', [])
        
        if not shots:
            return f"**Video Analysis: {video_name}**\n\nNo shots detected in this {duration:.1f} second video. For best results, ensure the full body and stick are visible, keep the camera steady, and film 10-15 shooting reps."
        
        # Build readable analysis text
        analysis_lines = [
            f"**Video Analysis: {video_name}**",
            f"Duration: {duration:.1f} seconds | FPS: {fps} | Shots detected: {len(shots)}",
            ""
        ]
        
        for i, shot in enumerate(shots, 1):
            shot_time = shot.get('shot_time_sec', 0)
            analysis_lines.append(f"**Shot {i}: {shot_time:.1f}s**")
            
            # NEW DATA FORMAT - Head position metrics
            head_pos = shot.get('head_position', {})
            if head_pos.get('head_up_score') is not None:
                head_score = head_pos['head_up_score']
                analysis_lines.append(f"**head position:** head {_score_to_category(head_score)} ({head_score:.0f}/100)")
            
            if head_pos.get('eyes_forward_score') is not None:
                eyes_score = head_pos['eyes_forward_score']
                analysis_lines.append(f"**eyes focused:** {_score_to_category(eyes_score)} ({eyes_score:.0f}/100)")
            
            # NEW: Wrist extension analysis
            wrist_ext = shot.get('wrist_extension', {})
            if wrist_ext.get('left_wrist_extension_score') is not None:
                left_score = wrist_ext['left_wrist_extension_score']
                analysis_lines.append(f"**left wrist extension:** {_score_to_category(left_score)} ({left_score:.0f}/100)")
            
            if wrist_ext.get('right_wrist_extension_score') is not None:
                right_score = wrist_ext['right_wrist_extension_score']
                analysis_lines.append(f"**right wrist extension:** {_score_to_category(right_score)} ({right_score:.0f}/100)")
            
            if wrist_ext.get('follow_through_score') is not None:
                follow_score = wrist_ext['follow_through_score']
                analysis_lines.append(f"**follow-through:** {_score_to_category(follow_score)} ({follow_score:.0f}/100)")
            
            # NEW: Hip rotation power analysis
            hip_rotation = shot.get('hip_rotation_power', {})
            if hip_rotation.get('max_rotation_speed') is not None:
                hip_speed = hip_rotation['max_rotation_speed']
                hip_angle = hip_rotation.get('rotation_angle_change', 0)
                analysis_lines.append(f"**hip rotation power:** {_score_to_category(hip_speed * 2)} ({hip_speed:.1f} speed, {hip_angle:.1f}° change)")
            
            # NEW: Front knee bend (direct access)
            if shot.get('front_knee_bend_deg') is not None:
                knee_angle = shot['front_knee_bend_deg']
                analysis_lines.append(f"**front knee bend:** {knee_angle:.0f}° ({_knee_angle_category(knee_angle)})")
            
            # NEW: Back leg drive analysis
            back_leg_drive = shot.get('back_leg_drive', {})
            if back_leg_drive.get('max_extension') is not None:
                back_extension = back_leg_drive['max_extension']
                back_angle = 180.0 - (back_extension / 10.0) if back_extension > 0 else 180.0  # Rough conversion
                analysis_lines.append(f"**back leg drive:** {back_extension:.1f} extension ({_leg_extension_category(back_angle)})")
            
            # NEW: Body stability analysis
            body_stability = shot.get('body_stability', {})
            if body_stability.get('stability_score') is not None:
                stability_score = body_stability['stability_score']
                analysis_lines.append(f"**body stability:** {_score_to_category(stability_score * 100)} ({stability_score:.2f})")
            
            # NEW: Weight transfer analysis
            weight_transfer = shot.get('weight_transfer', {})
            if weight_transfer.get('max_transfer_speed') is not None:
                transfer_speed = weight_transfer['max_transfer_speed']
                transfer_distance = weight_transfer.get('weight_shift_distance', 0)
                analysis_lines.append(f"**weight transfer:** {transfer_speed:.3f} speed, {transfer_distance:.3f} distance")
            
            # NEW: Torso rotation analysis
            torso_rotation = shot.get('torso_rotation', {})
            if torso_rotation.get('shoulder_rotation') is not None:
                shoulder_rot = torso_rotation['shoulder_rotation']
                hip_rot = torso_rotation.get('hip_rotation', 0)
                analysis_lines.append(f"**torso rotation:** shoulder {shoulder_rot:.1f}°, hip {hip_rot:.1f}°")
            
            
            analysis_lines.append("")  # Blank line between shots
        
        return "\n".join(analysis_lines)
        
    except Exception as exc:
        # Fallback to string representation if parsing fails
        return f"Raw analysis data (parsing error: {exc}):\n{str(raw_analysis)}"


def _score_to_category(score: float) -> str:
    """Convert 0-100 score to readable category."""
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 50:
        return "fair"
    else:
        return "needs work"


def _knee_angle_category(angle: float) -> str:
    """Convert knee angle to readable category."""
    if angle <= 100:
        return "excellent power position"
    elif angle <= 110:
        return "good bend"
    elif angle <= 130:
        return "moderate bend"
    else:
        return "too straight"


def _leg_extension_category(angle: float) -> str:
    """Convert leg extension angle to readable category."""
    if angle >= 160:
        return "excellent extension"
    elif angle >= 140:
        return "good extension"
    elif angle >= 120:
        return "moderate extension"
    else:
        return "limited extension"


def load_sample_analysis(filepath: str) -> str:
    """Load analysis data from a JSON file for testing."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if this is raw pose analysis data
        if isinstance(data, dict) and 'shots' in data and 'video' in data:
            # This is raw pose_extraction_shooting_drills.py output
            return parse_raw_pose_analysis(data)
        
        # Extract the data_analysis field if it's a processed analysis result
        if isinstance(data, dict) and 'analysis' in data:
            return data['analysis'].get('data_analysis', str(data))
        elif isinstance(data, dict) and 'data_analysis' in data:
            return data['data_analysis']
        else:
            return str(data)
    except Exception as exc:
        raise RuntimeError(f"Failed to load analysis file {filepath}: {exc}") from exc


def main():
    """Command-line interface for testing OpenIce agent."""
    parser = argparse.ArgumentParser(description="OpenIce Hockey Shooting Coach")
    parser.add_argument('--analysis-file', type=str, 
                       help='Path to analysis JSON file')
    parser.add_argument('--analysis-text', type=str,
                       help='Direct analysis text input')
    parser.add_argument('--question', type=str, required=True,
                       help='Question to ask about the shooting technique')
    parser.add_argument('--session-id', type=str,
                       help='Existing session ID to continue conversation')
    parser.add_argument('--user-id', type=str, default='test_user',
                       help='User identifier for the session')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode - validate input without calling Gemini API')
    
    args = parser.parse_args()
    
    # Test mode - validate input without calling Gemini API
    if args.test_mode:
        print("🧪 OpenIce Test Mode")
        if args.analysis_file:
            try:
                analysis_data = load_sample_analysis(args.analysis_file)
                print(f"✅ Successfully loaded analysis file: {args.analysis_file}")
                print(f"📊 Analysis data preview: {analysis_data[:200]}...")
                print(f"❓ Question: {args.question}")
                print("✅ Test mode validation passed! Ready for API integration.")
                return
            except Exception as e:
                print(f"❌ Error loading analysis file: {e}")
                sys.exit(1)
        elif args.analysis_text:
            print(f"✅ Analysis text provided: {args.analysis_text[:100]}...")
            print(f"❓ Question: {args.question}")
            print("✅ Test mode validation passed!")
            return
        else:
            print("❌ Test mode requires --analysis-file or --analysis-text")
            sys.exit(1)
    
    try:
        agent = OpenIceAgent()
        
        if args.session_id:
            # Continue existing conversation
            try:
                result = agent.ask_question(args.session_id, args.question)
                print(f"\n🏒 OpenIce Response:")
                print(f"{result['answer']}")
                
                if result['search_queries']:
                    print(f"\n🔍 Searched for: {', '.join(result['search_queries'])}")
                if result['sources']:
                    print(f"📚 Sources: {', '.join(result['sources'][:3])}")
                    
            except ValueError as e:
                print(f"❌ Error: {e}")
                sys.exit(1)
        else:
            # Start new conversation
            if args.analysis_file:
                analysis_data = load_sample_analysis(args.analysis_file)
            elif args.analysis_text:
                analysis_data = args.analysis_text
            else:
                print("❌ Error: Must provide either --analysis-file or --analysis-text for new session")
                sys.exit(1)
            
            print(f"🚀 Starting new OpenIce session...")
            session_id = agent.create_chat_session(analysis_data, args.user_id)
            print(f"📱 Session ID: {session_id}")
            
            result = agent.ask_question(session_id, args.question)
            print(f"\n🏒 OpenIce Response:")
            print(f"{result['answer']}")
            
            if result['search_queries']:
                print(f"\n🔍 Searched for: {', '.join(result['search_queries'])}")
            if result['sources']:
                print(f"📚 Sources: {', '.join(result['sources'][:3])}")
            
            print(f"\n💬 Continue conversation with: --session-id {session_id}")
            
    except Exception as exc:
        print(f"❌ OpenIce Error: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
