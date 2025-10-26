# services/session_service.py
from datetime import datetime
import json

class SessionService:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        """Get session data"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id, data):
        """Update session data"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created_at': datetime.now(),
                'step': 1
            }
        
        self.sessions[session_id].update(data)
        self.sessions[session_id]['last_update'] = datetime.now()
    
    def clear_session(self, session_id):
        """Clear session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_all_sessions(self):
        """Get all sessions"""
        return self.sessions