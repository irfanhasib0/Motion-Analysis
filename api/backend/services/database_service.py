import sqlite3
import json
import os
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to project root database path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            db_path = os.path.join(project_root, "nvr_database.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = os.path.abspath(db_path)
        logger.info(f"Using database at: {self.db_path}")
        self.init_database()
        
        # Load camera configurations from YAML
        self.load_camera_config()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    camera_type TEXT DEFAULT 'webcam',
                    fps INTEGER DEFAULT 30,
                    resolution TEXT DEFAULT '1920x1080',
                    status TEXT DEFAULT 'offline',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_active BOOLEAN DEFAULT 0,
                    processing_type TEXT,
                    processing_params TEXT
                )
            ''')
            
            # Ensure camera_type column exists (migration without exceptions)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(cameras)")
            columns = [row[1] for row in cursor.fetchall()]
            if "camera_type" not in columns:
                conn.execute('ALTER TABLE cameras ADD COLUMN camera_type TEXT DEFAULT "webcam"')
                logger.info("Added camera_type column to existing cameras table")
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration INTEGER,
                    file_size INTEGER,
                    status TEXT DEFAULT 'recording',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (camera_id) REFERENCES cameras (id)
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_recordings_camera_id ON recordings(camera_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_recordings_start_time ON recordings(start_time)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cameras_status ON cameras(status)')
            
            conn.commit()
            
            # Log database status
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cameras")
            camera_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM recordings")  
            recording_count = cursor.fetchone()[0]
            
            logger.info(f"Database initialized at {self.db_path}")
            logger.info(f"Found {camera_count} cameras and {recording_count} recordings in database")

    def load_camera_config(self):
        """Load camera configurations from YAML file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.dirname(current_dir)  # Go up from services to api directory
        config_path = os.path.join(api_dir, "config", "cameras.yaml")
        
        if not os.path.exists(config_path):
            logger.info(f"Camera config file not found at {config_path}, skipping camera initialization")
            return
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        if 'cameras' not in config:
            logger.warning("No 'cameras' section found in config file")
            return
            
        cameras_loaded = 0
        cameras_updated = 0
        
        for camera_config in config['cameras']:
            # Validate required fields
            required_fields = ['id', 'name', 'source']
            missing_fields = [field for field in required_fields if field not in camera_config]
            if missing_fields:
                logger.warning(f"Skipping camera config missing required fields: {missing_fields}")
                continue
            
            # Set defaults for optional fields
            camera_data = {
                'id': camera_config['id'],
                'name': camera_config['name'],
                'source': camera_config['source'],
                'camera_type': camera_config.get('type', 'webcam'),
                'fps': camera_config.get('fps', 30),
                'resolution': camera_config.get('resolution', '1920x1080'),
                'status': 'offline',  # Always start as offline
                'processing_params': json.dumps({
                    'description': camera_config.get('description', ''),
                    'location': camera_config.get('location', '')
                })
            }
            
            # Check if camera already exists
            existing_camera = self.get_camera(camera_config['id'])
            
            if existing_camera:
                # Update existing camera with config values
                self.update_camera(camera_config['id'], camera_data)
                cameras_updated += 1
                logger.info(f"Updated camera from config: {camera_data['name']} ({camera_data['id']})")
            else:
                # Create new camera
                self.create_camera(camera_data)
                cameras_loaded += 1
                logger.info(f"Created camera from config: {camera_data['name']} ({camera_data['id']})")
                
        logger.info(f"Camera config loaded: {cameras_loaded} new cameras created, {cameras_updated} existing cameras updated")
    
    
    # Camera operations
    def create_camera(self, camera_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new camera record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cameras (id, name, source, camera_type, fps, resolution, status, processing_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_data['id'],
                camera_data['name'],
                camera_data['source'],
                camera_data.get('camera_type', 'webcam'),
                camera_data.get('fps', 30),
                camera_data.get('resolution', '1920x1080'),
                camera_data.get('status', 'offline'),
                json.dumps(camera_data.get('processing_params', {}))
            ))
            
            conn.commit()
            return self.get_camera(camera_data['id'])
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get a camera by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,))
            row = cursor.fetchone()
            
            if row:
                camera = dict(row)
                camera['processing_params'] = json.loads(camera.get('processing_params', '{}'))
                camera['processing_active'] = bool(camera['processing_active'])
                return camera
            return None
    
    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """Get all cameras"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM cameras ORDER BY created_at')
            rows = cursor.fetchall()
            
            cameras = []
            for row in rows:
                camera = dict(row)
                camera['processing_params'] = json.loads(camera.get('processing_params', '{}'))
                camera['processing_active'] = bool(camera['processing_active'])
                cameras.append(camera)
            
            return cameras
    
    def update_camera(self, camera_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a camera record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = []
            update_values = []
            
            for key, value in updates.items():
                if key == 'processing_params':
                    update_fields.append(f"{key} = ?")
                    update_values.append(json.dumps(value))
                elif key == 'processing_active':
                    update_fields.append(f"{key} = ?")
                    update_values.append(1 if value else 0)
                else:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)
            
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            update_values.append(camera_id)
            
            query = f"UPDATE cameras SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, update_values)
            
            conn.commit()
            return self.get_camera(camera_id)
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cameras WHERE id = ?', (camera_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
    
    # Recording operations
    def create_recording(self, recording_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new recording record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO recordings (id, camera_id, file_path, start_time, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                recording_data['id'],
                recording_data['camera_id'],
                recording_data['file_path'],
                recording_data['start_time'],
                recording_data.get('status', 'recording')
            ))
            
            conn.commit()
            return self.get_recording(recording_data['id'])
    
    def get_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get a recording by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,))
            row = cursor.fetchone()
            
            return dict(row) if row else None
    
    def get_recordings_by_camera(self, camera_id: str) -> List[Dict[str, Any]]:
        """Get all recordings for a specific camera"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM recordings 
                WHERE camera_id = ? 
                ORDER BY start_time DESC
            ''', (camera_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_recordings(self) -> List[Dict[str, Any]]:
        """Get all recordings"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT r.*, c.name as camera_name
                FROM recordings r
                LEFT JOIN cameras c ON r.camera_id = c.id
                ORDER BY r.start_time DESC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_recording(self, recording_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a recording record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build dynamic update query
            update_fields = []
            update_values = []
            
            for key, value in updates.items():
                update_fields.append(f"{key} = ?")
                update_values.append(value)
            
            update_values.append(recording_id)
            
            query = f"UPDATE recordings SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, update_values)
            
            conn.commit()
            return self.get_recording(recording_id)
    
    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM recordings WHERE id = ?', (recording_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
    
    def get_active_recordings(self) -> List[Dict[str, Any]]:
        """Get all currently active recordings"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT r.*, c.name as camera_name
                FROM recordings r
                LEFT JOIN cameras c ON r.camera_id = c.id
                WHERE r.status = 'recording'
            ''')
            
            return [dict(row) for row in cursor.fetchall()]