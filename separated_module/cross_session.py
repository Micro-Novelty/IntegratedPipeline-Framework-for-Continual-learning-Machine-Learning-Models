"""CrossSessionAutomation: automation that spans training sessions."""

import glob
import json
import socket
from datetime import datetime


# Cross-session automation module that allows exporting and importing of sessions, syncing with another device, and listing available sessions for better management and continuity of work across different environments.
class CrossSessionAutomation:
    """Export, import and sync pipeline memories between sessions (JSON files, or over TLS to another device)."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_session(self, session_name=None):
        """Write the pipeline's memories and metadata to a JSON file and return its filename."""
        if session_name is None:
            session_name = f"session_{self.session_id}"
        
        session_data = {
            'session_id': self.session_id,
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'memories': self.pipeline.memory.copy(),
        }
        
        filename = f"{session_name}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
        
        print(f"💾 Session exported to: {filename}")
        return filename
    
    def import_session(self, filename):
        """Load a session JSON file and merge its memories into the pipeline."""
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        print(f"\n📥 Importing session: {session_data['session_name']}")
        print(f"   Created: {session_data['timestamp']}")
        print(f"   Memories: {len(session_data['memories'])}")
        
        # Merge memories
        for key, value in session_data['memories'].items():
            if key not in self.pipeline.memory:
                self.pipeline.memory[key] = value
        
        print(f"✅ Session imported! Total memories: {len(self.pipeline.memory)}")
    
    def sync_with_another_device(self, device_ip, port=5000):
        """Export the current session and send it to another device over a TLS socket."""
        # Export current session
        temp_file = self.export_session(f"sync_{self.session_id}")
        
        try:
            with self.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.connect((device_ip, port))
                with open(temp_file, 'rb') as f:
                    s.sendall(f.read())
                print(f"📡 Synced to {device_ip} || {port}")
                print('🚀 Succesfully sync and export memory session to another device! ')                 
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            pass
        

    
    def list_sessions(self, name):
        """Print the saved session files whose names start with `name`, with their memory counts."""
        sessions = glob.glob(f"{name}*.json")
        
        print(f"\n📚 Available Sessions: {sessions}")
        if sessions:
            for session in sessions:
                with open(session, 'r') as f:
                    data = json.load(f)
                    print(f"   • {session}: {data['session_name']} ({len(data['memories'])} memories)")

        else:          
            print('[-] No available sessions! ')
        
        return sessions
