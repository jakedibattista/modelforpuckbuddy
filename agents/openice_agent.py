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
        system_prompt = f"""You are OpenIce, an expert hockey shooting coach with deep knowledge of NHL players and modern training techniques.

PLAYER'S TECHNICAL ANALYSIS:
{analysis_data}

Your role:
- Provide specific, actionable coaching advice
- Reference the technical data and specific shot timestamps when relevant
- Use web search to find current information about NHL players, techniques, and drills
- Give personalized recommendations based on the player's actual data
- Be encouraging but honest about areas that need improvement

Always reference specific shots by timestamp (e.g., "In your shot at 00:15") when providing feedback about technique.

Guidelines:
- Keep responses conversational but informative
- Include specific measurements and comparisons when helpful
- Suggest concrete drills and practice methods
- When comparing to NHL players, cite recent technique analysis
"""

        try:
            # Create chat session with Google Search enabled
            chat = self.client.chats.create(
                model='gemini-2.5-flash',
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    max_output_tokens=1024
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
            # Send the question to the chat
            response = chat.send_message(question)
            
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
            
            return {
                'answer': response.text,
                'search_queries': search_queries,
                'sources': sources,
                'session_id': session_id,
                'message_count': session['message_count']
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


def load_sample_analysis(filepath: str) -> str:
    """Load analysis data from a JSON file for testing."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract the data_analysis field if it's a full analysis result
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
