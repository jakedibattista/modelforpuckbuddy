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
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        f"Google GenAI library not installed: {exc}\n"
        "Install with: pip install google-genai"
    ) from exc

# Configure logging
logger = logging.getLogger(__name__)


# Elite NHL Players Database - 25 distinct shooting styles mapped to biomechanical attributes
# Each player profile maps to specific metrics from pose analysis
NHL_PLAYER_PROFILES = {
    "alex_ovechkin": {
        "name": "Alex Ovechkin",
        "primary_strength": "One-timer power from the circle",
        "key_mechanics": "Explosive hip rotation, perfect weight transfer, wide base",
        "best_for": ["hip_rotation", "weight_transfer", "power_generation"],
        "signature_move": "The Ovi spot one-timer - opens hips fully to the net",
        "era": "Current"
    },
    "auston_matthews": {
        "name": "Auston Matthews",
        "primary_strength": "Quick release wrist shot",
        "key_mechanics": "Minimal windup, exceptional wrist extension, tight puck control",
        "best_for": ["wrist_extension", "quick_release", "head_position"],
        "signature_move": "Ultra-fast wrist shot with disguised release point",
        "era": "Current"
    },
    "connor_mcdavid": {
        "name": "Connor McDavid",
        "primary_strength": "Lightning-fast shot in motion",
        "key_mechanics": "Maintains balance at high speed, quick hands, excellent body control",
        "best_for": ["body_stability", "quick_release", "in_stride_shooting"],
        "signature_move": "Full-speed snapshot with perfect balance",
        "era": "Current"
    },
    "sidney_crosby": {
        "name": "Sidney Crosby",
        "primary_strength": "Accurate shots from any angle",
        "key_mechanics": "Head always up, reads goalies, deceptive release",
        "best_for": ["head_position", "accuracy", "deceptive_release"],
        "signature_move": "Backhand roof shot - keeps head up, picks corners",
        "era": "Current"
    },
    "steven_stamkos": {
        "name": "Steven Stamkos",
        "primary_strength": "Perfect one-timer timing",
        "key_mechanics": "Wide stance, perfect weight shift, follows through completely",
        "best_for": ["weight_transfer", "follow_through", "one_timer_mechanics"],
        "signature_move": "PP one-timer - loads weight on back leg, explodes through puck",
        "era": "Current"
    },
    "patrick_kane": {
        "name": "Patrick Kane",
        "primary_strength": "Deceptive release angles",
        "key_mechanics": "Constantly changes shot angle, exceptional wrist flexibility",
        "best_for": ["wrist_extension", "deceptive_release", "quick_hands"],
        "signature_move": "Changes angle mid-shot, goalies never set",
        "era": "Current"
    },
    "nathan_mackinnon": {
        "name": "Nathan MacKinnon",
        "primary_strength": "Power shot off the rush",
        "key_mechanics": "Drives through puck with full body, explosive hip turn",
        "best_for": ["hip_rotation", "power_generation", "torso_rotation"],
        "signature_move": "Rushes wide, cuts to middle, rips shot far side",
        "era": "Current"
    },
    "cale_makar": {
        "name": "Cale Makar",
        "primary_strength": "Accurate shot while skating",
        "key_mechanics": "Perfect balance on edges, keeps upper body stable",
        "best_for": ["body_stability", "knee_bend", "in_stride_shooting"],
        "signature_move": "Skates laterally, maintains balance, picks corners",
        "era": "Current"
    },
    "joe_pavelski": {
        "name": "Joe Pavelski",
        "primary_strength": "Net-front tips and quick release",
        "key_mechanics": "Quick hands in tight, exceptional eye-hand coordination",
        "best_for": ["quick_release", "hand_positioning", "net_front_play"],
        "signature_move": "Deflections and tips - incredible hand-eye",
        "era": "Current"
    },
    "mark_stone": {
        "name": "Mark Stone",
        "primary_strength": "Surgical precision placement",
        "key_mechanics": "Patient release, picks spots, head always tracking goalie",
        "best_for": ["head_position", "accuracy", "shot_placement"],
        "signature_move": "Waits for goalie to move, places shot in opening",
        "era": "Current"
    },
    "brad_marchand": {
        "name": "Brad Marchand",
        "primary_strength": "Quick snapshot in stride",
        "key_mechanics": "Minimal windup, shoots while moving, great balance",
        "best_for": ["quick_release", "body_stability", "in_stride_shooting"],
        "signature_move": "Snap shot off the rush without breaking stride",
        "era": "Current"
    },
    "leon_draisaitl": {
        "name": "Leon Draisaitl",
        "primary_strength": "Deceptive one-timer angles",
        "key_mechanics": "Changes angle at last second, powerful lower body",
        "best_for": ["deceptive_release", "weight_transfer", "one_timer_mechanics"],
        "signature_move": "One-timer from unexpected angles",
        "era": "Current"
    },
    "kirill_kaprizov": {
        "name": "Kirill Kaprizov",
        "primary_strength": "Creative shot angles",
        "key_mechanics": "Exceptional flexibility, can shoot from any position",
        "best_for": ["wrist_extension", "deceptive_release", "creativity"],
        "signature_move": "Finds shooting lanes no one else sees",
        "era": "Current"
    },
    "wayne_gretzky": {
        "name": "Wayne Gretzky",
        "primary_strength": "Perfect shot placement and timing",
        "key_mechanics": "Head always up reading play, patient release",
        "best_for": ["head_position", "accuracy", "shot_placement"],
        "signature_move": "Never rushed, always picked the perfect spot",
        "era": "Legend"
    },
    "mario_lemieux": {
        "name": "Mario Lemieux",
        "primary_strength": "Power and finesse combined",
        "key_mechanics": "Long reach, powerful wrists, could shoot from anywhere",
        "best_for": ["wrist_extension", "power_generation", "versatility"],
        "signature_move": "One-hand shots, backhand snipes, unstoppable",
        "era": "Legend"
    },
    "brett_hull": {
        "name": "Brett Hull",
        "primary_strength": "One-timer specialist",
        "key_mechanics": "Perfect weight shift, followed through completely",
        "best_for": ["weight_transfer", "follow_through", "one_timer_mechanics"],
        "signature_move": "Set up in slot, one-timer top corner",
        "era": "Legend"
    },
    "pavel_datsyuk": {
        "name": "Pavel Datsyuk",
        "primary_strength": "Master of deception",
        "key_mechanics": "Constantly fakes, incredible hands, deceives goalies",
        "best_for": ["deceptive_release", "wrist_extension", "creativity"],
        "signature_move": "Makes goalies commit, shoots opposite direction",
        "era": "Legend"
    },
    "teemu_selanne": {
        "name": "Teemu Selänne",
        "primary_strength": "Shot off the rush",
        "key_mechanics": "Maintained speed, quick release, perfect balance",
        "best_for": ["in_stride_shooting", "quick_release", "body_stability"],
        "signature_move": "Full-speed wrist shot while cutting to net",
        "era": "Legend"
    },
    "mike_bossy": {
        "name": "Mike Bossy",
        "primary_strength": "Pure goal scorer's shot",
        "key_mechanics": "Quick release, perfect follow-through, always on target",
        "best_for": ["quick_release", "follow_through", "accuracy"],
        "signature_move": "Fastest release of his era, unstoppable in close",
        "era": "Legend"
    },
    "al_macinnis": {
        "name": "Al MacInnis",
        "primary_strength": "Hardest slap shot ever",
        "key_mechanics": "Full extension, explosive hip drive, perfect weight transfer",
        "best_for": ["power_generation", "weight_transfer", "back_leg_drive"],
        "signature_move": "Point shot that injured goalies",
        "era": "Legend"
    },
    "peter_forsberg": {
        "name": "Peter Forsberg",
        "primary_strength": "Power and balance combined",
        "key_mechanics": "Low knee bend, explosive through contact, strong base",
        "best_for": ["knee_bend", "body_stability", "power_generation"],
        "signature_move": "Could shoot while being checked, incredible balance",
        "era": "Legend"
    },
    "pavel_bure": {
        "name": "Pavel Bure",
        "primary_strength": "Speed shot specialist",
        "key_mechanics": "Shot at maximum speed, no windup needed",
        "best_for": ["quick_release", "in_stride_shooting", "minimal_windup"],
        "signature_move": "Breakaway specialist - shot while at top speed",
        "era": "Legend"
    },
    "jaromir_jagr": {
        "name": "Jaromír Jágr",
        "primary_strength": "Shield and shoot technique",
        "key_mechanics": "Powerful lower body, protected puck, quick release",
        "best_for": ["body_stability", "knee_bend", "puck_protection"],
        "signature_move": "Used body to shield, then quick wrister",
        "era": "Legend"
    },
    "alexander_mogilny": {
        "name": "Alexander Mogilny",
        "primary_strength": "Pure shooting talent",
        "key_mechanics": "Quick hands, accurate, could score from anywhere",
        "best_for": ["wrist_extension", "quick_release", "versatility"],
        "signature_move": "Lightning-fast wrist shot, goalies had no chance",
        "era": "Legend"
    },
    "luc_robitaille": {
        "name": "Luc Robitaille",
        "primary_strength": "Net-front specialist",
        "key_mechanics": "Found soft areas, quick release in traffic",
        "best_for": ["net_front_play", "quick_release", "positioning"],
        "signature_move": "Always in right spot, quick shot before goalie set",
        "era": "Legend"
    }
}


class OpenIceAgent:
    """
    Conversational AI coach for hockey shooting analysis.
    
    Uses Gemini's chat capabilities with Google Search integration to provide
    intelligent, context-aware responses about hockey technique.
    
    Features:
    - Context management for long conversations
    - Message summarization to prevent token limit issues
    - Session persistence and memory optimization
    - Error recovery for complex chat sessions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenIce agent with Gemini client."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        self.chat_sessions: Dict[str, Any] = {}
        
        # Context management settings
        self.max_context_messages = 10  # Keep last 10 messages in context
        self.max_tokens_per_request = 8000  # Conservative token limit
        self.summarization_threshold = 15  # Summarize when > 15 messages
        
        # Error handling settings
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
    
    def _analyze_shooting_strengths_weaknesses(self, analysis_data: str) -> Dict[str, List[str]]:
        """Extract strengths and weaknesses from analysis data."""
        strengths = []
        weaknesses = []
        
        # Parse analysis data to identify key metrics
        analysis_lower = analysis_data.lower()
        
        # Check for good scores/categories (strengths)
        if any(term in analysis_lower for term in ["excellent", "very good", "good"]):
            if "hip rotation" in analysis_lower and any(x in analysis_lower for x in ["excellent", "good"]):
                strengths.append("hip_rotation")
            if "wrist extension" in analysis_lower and any(x in analysis_lower for x in ["excellent", "good"]):
                strengths.append("wrist_extension")
            if "head position" in analysis_lower and any(x in analysis_lower for x in ["excellent", "good"]):
                strengths.append("head_position")
            if "body stability" in analysis_lower and any(x in analysis_lower for x in ["excellent", "good"]):
                strengths.append("body_stability")
            if "weight transfer" in analysis_lower and any(x in analysis_lower for x in ["excellent", "good"]):
                strengths.append("weight_transfer")
            if "knee bend" in analysis_lower and ("excellent" in analysis_lower or "good bend" in analysis_lower):
                strengths.append("knee_bend")
        
        # Check for poor scores/categories (weaknesses)
        if any(term in analysis_lower for term in ["needs work", "poor", "fair", "minimal"]):
            if "hip rotation" in analysis_lower and any(x in analysis_lower for x in ["needs work", "poor", "fair"]):
                weaknesses.append("hip_rotation")
            if "wrist" in analysis_lower and any(x in analysis_lower for x in ["needs work", "poor", "fair"]):
                weaknesses.append("wrist_extension")
            if "head" in analysis_lower and any(x in analysis_lower for x in ["needs work", "poor", "dropped"]):
                weaknesses.append("head_position")
            if "stability" in analysis_lower and any(x in analysis_lower for x in ["poor", "needs work"]):
                weaknesses.append("body_stability")
            if "weight transfer" in analysis_lower and any(x in analysis_lower for x in ["poor", "needs work"]):
                weaknesses.append("weight_transfer")
            if "knee" in analysis_lower and any(x in analysis_lower for x in ["too straight", "shallow", "needs work"]):
                weaknesses.append("knee_bend")
            if "follow through" in analysis_lower and any(x in analysis_lower for x in ["rushed", "poor"]):
                weaknesses.append("follow_through")
        
        return {"strengths": strengths, "weaknesses": weaknesses}
    
    def _match_players_to_profile(self, strengths: List[str], weaknesses: List[str]) -> str:
        """Match NHL players to user's specific strengths and weaknesses."""
        matched_players = []
        
        # Priority 1: Match players who excel in user's weakness areas (players to learn from)
        weakness_matches = []
        for player_id, profile in NHL_PLAYER_PROFILES.items():
            for weakness in weaknesses:
                if weakness in profile["best_for"]:
                    weakness_matches.append({
                        "player": profile["name"],
                        "reason": f"Work on {weakness.replace('_', ' ')} - {profile['primary_strength']}",
                        "technique": profile["key_mechanics"],
                        "signature": profile["signature_move"],
                        "priority": "improve"
                    })
                    break  # Only add each player once
        
        # Priority 2: Match players who share user's strengths (validation/style match)
        strength_matches = []
        for player_id, profile in NHL_PLAYER_PROFILES.items():
            for strength in strengths:
                if strength in profile["best_for"]:
                    strength_matches.append({
                        "player": profile["name"],
                        "reason": f"Similar strength in {strength.replace('_', ' ')} - {profile['primary_strength']}",
                        "technique": profile["key_mechanics"],
                        "signature": profile["signature_move"],
                        "priority": "model"
                    })
                    break  # Only add each player once
        
        # Build recommendation text
        recommendations = []
        
        # Add weakness-focused players (up to 2)
        if weakness_matches:
            recommendations.append("**PLAYERS TO STUDY (Areas to Improve):**")
            for i, match in enumerate(weakness_matches[:2], 1):
                recommendations.append(
                    f"{i}. **{match['player']}** - {match['reason']}\n"
                    f"   • Key mechanics: {match['technique']}\n"
                    f"   • Signature move: {match['signature']}"
                )
        
        # Add strength-matched players (up to 1)
        if strength_matches:
            recommendations.append("\n**PLAYERS WITH SIMILAR STRENGTHS (Your Style):**")
            match = strength_matches[0]  # Just show the best match
            recommendations.append(
                f"• **{match['player']}** - {match['reason']}\n"
                f"  Key mechanics: {match['technique']}"
            )
        
        # Fallback if no matches found
        if not recommendations:
            recommendations.append(
                "**VERSATILE PLAYERS TO STUDY:**\n"
                "• **Sidney Crosby** - All-around excellence, head always up\n"
                "• **Auston Matthews** - Quick release fundamentals\n"
                "• **Connor McDavid** - Balance and body control"
            )
        
        return "\n".join(recommendations)
    
    def _get_player_recommendations(self, analysis_data: str) -> str:
        """Generate intelligent player recommendations based on analysis data."""
        # Analyze the user's shooting profile
        profile = self._analyze_shooting_strengths_weaknesses(analysis_data)
        
        # Match players to their profile
        player_matches = self._match_players_to_profile(
            profile["strengths"], 
            profile["weaknesses"]
        )
        
        return f"""
NHL PLAYER RECOMMENDATIONS (Matched to Your Profile):

{player_matches}

COACHING APPROACH:
- When user asks about player comparisons, reference the players above
- Explain SPECIFIC techniques from these players that match their needs
- Focus on players in "Areas to Improve" for development advice
- Reference "Similar Strengths" players to validate what they're doing well
- Always explain WHY that player's technique would help them

Example: "Your wrist extension needs work - study Auston Matthews' technique: he keeps minimal windup and uses exceptional wrist snap. Notice how he keeps the puck close and uses his bottom hand to generate power through wrist roll."
"""
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _summarize_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize old conversation history to maintain context without token overflow."""
        if not messages:
            return ""
        
        # Extract key points from conversation
        key_points = []
        for msg in messages:
            if msg.get('role') == 'user':
                key_points.append(f"User asked: {msg.get('content', '')[:100]}...")
            elif msg.get('role') == 'assistant':
                key_points.append(f"Coach advised: {msg.get('content', '')[:100]}...")
        
        summary = "Previous conversation highlights:\n" + "\n".join(key_points[-5:])  # Last 5 exchanges
        return summary
    
    def _manage_context(self, session: Dict[str, Any]) -> bool:
        """Manage conversation context to prevent token limit issues."""
        try:
            # Check if we need to summarize
            message_count = session.get('message_count', 0)
            if message_count < self.summarization_threshold:
                return True
            
            # Get current chat history
            chat = session['chat']
            
            # Create a new chat session with summarized context
            logger.info(f"Summarizing conversation for session {session.get('session_id', 'unknown')}")
            
            # Get conversation summary
            summary_prompt = """Please provide a concise summary of our conversation so far, focusing on:
1. The main shooting issues we've discussed
2. Key advice given
3. Current focus areas for improvement
4. Any NHL players mentioned as references

Keep it under 200 words and maintain the coaching context."""
            
            try:
                summary_response = chat.send_message(summary_prompt)
                conversation_summary = summary_response.text.strip()
            except Exception as e:
                logger.warning(f"Failed to get conversation summary: {e}")
                conversation_summary = "Previous conversation covered shooting technique improvements."
            
            # Get fresh player recommendations for context reset
            player_recommendations = self._get_player_recommendations(session['analysis_data'])
            
            # Create optimized new chat with summarized context
            new_system_prompt = f"""You are OpenIce, a direct hockey shooting coach. Give specific, actionable feedback under 150 words.

PLAYER DATA:
{session['analysis_data']}

CONTEXT: {conversation_summary}

{player_recommendations}

RULES: Cite specific scores/timestamps. One improvement area. Specific drills only.
STYLE: Direct + encouraging. Data-driven.
Be specific, be brief, be helpful.
"""
            
            # Create new chat session
            new_chat = self.client.chats.create(
                model='gemini-2.0-flash-exp-001',
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    max_output_tokens=10000,
                    temperature=0.4,
                    top_p=0.8
                )
            )
            
            # Initialize with new context
            new_chat.send_message(new_system_prompt)
            
            # Update session with new chat
            session['chat'] = new_chat
            session['message_count'] = 0  # Reset counter
            session['last_activity'] = datetime.now()
            
            logger.info("Successfully created new chat context")
            return True
            
        except Exception as e:
            logger.error(f"Failed to manage context: {e}")
            return False
    
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
        
        # Get intelligent player recommendations based on analysis
        player_recommendations = self._get_player_recommendations(analysis_data)
        
        # Create optimized system prompt with analysis context
        system_prompt = f"""You are OpenIce, a direct hockey shooting coach. Give specific, actionable feedback under 150 words.

PLAYER DATA:
{analysis_data}

{player_recommendations}

RULES:
1. Always cite specific scores, timestamps, and angles from their data
2. One main improvement area per response
3. Give specific drills, not generic advice

STYLE: Direct + encouraging ("Your 72/100 head position needs work at 00:08 - try the wall drill")
SCORES: 80-100 excellent | 60-79 good | <60 needs work

NHL COMPARISONS: Use matched players above. Explain WHAT technique to copy.

Be specific, be brief, be helpful.
"""

        try:
            # Create chat session with Google Search enabled and optimized for concise responses
            # Increased max_output_tokens to better support longer conversations
            chat = self.client.chats.create(
                model='gemini-flash-latest',
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    max_output_tokens=2000,  # Increased from 1000 to support longer, complex chats
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
        Ask a question in an existing chat session with context management.
        
        Args:
            session_id: Chat session identifier
            question: User's question about their shooting technique
            
        Returns:
            Dictionary containing response and metadata
        """
        if session_id not in self.chat_sessions:
            raise ValueError(f"Chat session {session_id} not found")
        
        session = self.chat_sessions[session_id]
        
        try:
            # Manage context before processing the question
            if not self._manage_context(session):
                logger.warning("Context management failed, proceeding with current context")
            
            chat = session['chat']
            
            # Enhance question with context hints for more focused responses
            enhanced_question = self._enhance_question_with_context(question, session)
            
            # Estimate token usage before sending
            estimated_tokens = self._estimate_tokens(enhanced_question)
            if estimated_tokens > self.max_tokens_per_request:
                logger.warning(f"Question too long ({estimated_tokens} tokens), truncating")
                enhanced_question = enhanced_question[:self.max_tokens_per_request * 4] + "..."
            
            # Send the enhanced question to the chat with enhanced retry logic
            response = None
            for attempt in range(self.max_retries):
                try:
                    response = chat.send_message(enhanced_question)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    
                    if attempt == self.max_retries - 1:
                        # Final attempt failed
                        logger.error(f"All retry attempts failed for session {session_id}")
                        raise RuntimeError(f"Failed to process question after {self.max_retries} attempts. This may be due to conversation length. Try starting a new session.") from e
                    
                    # Handle 500 errors and context issues by recreating the session
                    if "500" in error_str or "context" in error_str or "limit" in error_str:
                        logger.info(f"Detected context/500 error, attempting to manage context and retry")
                        if not self._manage_context(session):
                            logger.warning("Context management failed during retry")
                        chat = session['chat']
                    
                    # Wait before retrying
                    import time
                    time.sleep(self.retry_delay * (attempt + 1))
            
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
                except Exception as e:
                    logger.warning(f"Failed to shorten response: {e}")
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
            logger.error(f"Failed to process question for session {session_id}: {exc}")
            # Try to recover by creating a fresh context
            try:
                logger.info("Attempting to recover session with fresh context")
                if self._manage_context(session):
                    # Retry once with fresh context
                    return self.ask_question(session_id, question)
            except Exception as recovery_exc:
                logger.error(f"Recovery failed: {recovery_exc}")
            
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
            'message_count': session['message_count'],
            'context_managed': session.get('message_count', 0) >= self.summarization_threshold
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all active sessions."""
        total_sessions = len(self.chat_sessions)
        total_messages = sum(session.get('message_count', 0) for session in self.chat_sessions.values())
        long_conversations = sum(1 for session in self.chat_sessions.values() 
                               if session.get('message_count', 0) >= self.summarization_threshold)
        
        return {
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'long_conversations': long_conversations,
            'average_messages_per_session': total_messages / total_sessions if total_sessions > 0 else 0
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old chat sessions to manage memory."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            sid for sid, session in self.chat_sessions.items()
            if session['last_activity'] < cutoff
        ]
        
        for session_id in old_sessions:
            try:
                # Log session stats before cleanup
                session = self.chat_sessions[session_id]
                logger.info(f"Cleaning up session {session_id}: {session.get('message_count', 0)} messages")
                del self.chat_sessions[session_id]
            except Exception as e:
                logger.warning(f"Error cleaning up session {session_id}: {e}")
        
        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        return len(old_sessions)
    
    def force_context_reset(self, session_id: str) -> bool:
        """Force a context reset for a specific session."""
        if session_id not in self.chat_sessions:
            return False
        
        session = self.chat_sessions[session_id]
        return self._manage_context(session)


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
