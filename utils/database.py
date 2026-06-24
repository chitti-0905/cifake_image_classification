"""
SQLite Database Management
Handles user authentication, prediction history, and data persistence
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database.db')


class Database:
    """SQLite Database handler for CIFAKE"""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self.ensure_database_exists()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image_name TEXT NOT NULL,
                    image_path TEXT,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    interpretation TEXT,
                    heatmap_path TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            conn.commit()
            print("✓ Database initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing database: {e}")
        finally:
            conn.close()
    
    # ============ USER OPERATIONS ============
    
    def create_user(self, username, email, password_hash):
        """Create new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            user_id = cursor.lastrowid
            return {'id': user_id, 'username': username, 'email': email}
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return {'error': 'Username already exists'}
            elif 'email' in str(e):
                return {'error': 'Email already exists'}
            else:
                return {'error': str(e)}
        finally:
            conn.close()
    
    def get_user_by_username(self, username):
        """Get user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            conn.close()
    
    def get_user_by_email(self, email):
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            conn.close()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            conn.close()
    
    def update_user(self, user_id, **kwargs):
        """Update user information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Build UPDATE query dynamically
            updates = []
            values = []
            for key, value in kwargs.items():
                if key in ['username', 'email', 'password', 'is_active']:
                    updates.append(f'{key} = ?')
                    values.append(value)
            
            if not updates:
                return {'error': 'No valid fields to update'}
            
            updates.append('updated_at = CURRENT_TIMESTAMP')
            values.append(user_id)
            
            query = f'UPDATE users SET {", ".join(updates)} WHERE id = ?'
            cursor.execute(query, values)
            conn.commit()
            
            return {'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def delete_user(self, user_id):
        """Delete user (soft delete)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE users SET is_active = 0 WHERE id = ?
            ''', (user_id,))
            conn.commit()
            return {'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    # ============ PREDICTION OPERATIONS ============
    
    def save_prediction(self, user_id, image_name, prediction, confidence, 
                       interpretation=None, image_path=None, heatmap_path=None, 
                       processing_time=None):
        """Save prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO predictions 
                (user_id, image_name, image_path, prediction, confidence, 
                 interpretation, heatmap_path, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, image_name, image_path, prediction, confidence, 
                  interpretation, heatmap_path, processing_time))
            
            conn.commit()
            prediction_id = cursor.lastrowid
            return {'id': prediction_id, 'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_user_predictions(self, user_id, limit=50, offset=0):
        """Get user's prediction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
            
            predictions = [dict(row) for row in cursor.fetchall()]
            return predictions
        finally:
            conn.close()
    
    def get_prediction_by_id(self, prediction_id, user_id):
        """Get specific prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE id = ? AND user_id = ?
            ''', (prediction_id, user_id))
            
            prediction = cursor.fetchone()
            return dict(prediction) if prediction else None
        finally:
            conn.close()
    
    def delete_prediction(self, prediction_id, user_id):
        """Delete prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                DELETE FROM predictions 
                WHERE id = ? AND user_id = ?
            ''', (prediction_id, user_id))
            
            conn.commit()
            return {'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_user_prediction_count(self, user_id):
        """Get total predictions for user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT COUNT(*) as count FROM predictions WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            return result['count'] if result else 0
        finally:
            conn.close()
    
    def get_user_statistics(self, user_id):
        """Get user prediction statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Total predictions
            cursor.execute('''
                SELECT COUNT(*) as total FROM predictions WHERE user_id = ?
            ''', (user_id,))
            total = cursor.fetchone()['total']
            
            # Real vs Fake counts
            cursor.execute('''
                SELECT prediction, COUNT(*) as count FROM predictions 
                WHERE user_id = ? 
                GROUP BY prediction
            ''', (user_id,))
            predictions = {row['prediction']: row['count'] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence) as avg_confidence FROM predictions WHERE user_id = ?
            ''', (user_id,))
            avg_confidence = cursor.fetchone()['avg_confidence'] or 0
            
            return {
                'total_predictions': total,
                'real_count': predictions.get('REAL', 0),
                'fake_count': predictions.get('FAKE', 0),
                'average_confidence': round(avg_confidence, 2)
            }
        finally:
            conn.close()
    
    # ============ SESSION OPERATIONS ============
    
    def create_session(self, user_id, session_token, ip_address=None, user_agent=None, expires_at=None):
        """Create user session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, ip_address, user_agent, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_token, ip_address, user_agent, expires_at))
            
            conn.commit()
            return {'success': True, 'session_token': session_token}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_session(self, session_token):
        """Get session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM sessions 
                WHERE session_token = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ''', (session_token,))
            
            session = cursor.fetchone()
            return dict(session) if session else None
        finally:
            conn.close()
    
    def delete_session(self, session_token):
        """Delete session (logout)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            return {'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def delete_user_sessions(self, user_id):
        """Delete all user sessions (logout all devices)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
            conn.commit()
            return {'success': True}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    # ============ STATISTICS ============
    
    def get_system_statistics(self):
        """Get system-wide statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Total users
            cursor.execute('SELECT COUNT(*) as count FROM users WHERE is_active = 1')
            total_users = cursor.fetchone()['count']
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) as count FROM predictions')
            total_predictions = cursor.fetchone()['count']
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) as avg FROM predictions')
            avg_confidence = cursor.fetchone()['avg'] or 0
            
            # Fake vs Real distribution
            cursor.execute('''
                SELECT prediction, COUNT(*) as count FROM predictions 
                GROUP BY prediction
            ''')
            distribution = {row['prediction']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_users': total_users,
                'total_predictions': total_predictions,
                'average_confidence': round(avg_confidence, 2),
                'distribution': distribution
            }
        finally:
            conn.close()


# Global database instance
db = Database()


# Convenience functions
def init_db():
    """Initialize database"""
    return db.ensure_database_exists()


def get_db():
    """Get database instance"""
    return db
