"""
Authentication utilities for user management
Handles password hashing, verification, and token generation
"""

import hashlib
import secrets
import re
from datetime import datetime, timedelta


class PasswordManager:
    """Handle password hashing and verification"""
    
    @staticmethod
    def hash_password(password):
        """
        Hash password using SHA-256 with salt
        In production, use bcrypt or argon2
        """
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}${password_hash}"
    
    @staticmethod
    def verify_password(password, hash_stored):
        """Verify password against stored hash"""
        try:
            salt, password_hash = hash_stored.split('$')
            new_hash = hashlib.sha256((salt + password).encode()).hexdigest()
            return new_hash == password_hash
        except Exception as e:
            print(f"Error verifying password: {e}")
            return False


class TokenManager:
    """Generate and validate tokens"""
    
    @staticmethod
    def generate_session_token(length=32):
        """Generate secure session token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_verification_token(length=32):
        """Generate email verification token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def get_token_expiry(hours=24):
        """Get token expiry time"""
        return (datetime.utcnow() + timedelta(hours=hours)).isoformat()


class Validator:
    """Validate user input"""
    
    @staticmethod
    def validate_username(username):
        """
        Validate username
        - 3-20 characters
        - Alphanumeric and underscore
        - Not starting with number
        """
        if not username:
            return False, "Username is required"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(username) > 20:
            return False, "Username must be at most 20 characters"
        
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', username):
            return False, "Username must contain only letters, numbers, and underscores"
        
        return True, None
    
    @staticmethod
    def validate_email(email):
        """Validate email format"""
        if not email:
            return False, "Email is required"
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return False, "Invalid email format"
        
        return True, None
    
    @staticmethod
    def validate_password(password):
        """
        Validate password strength
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        
        return True, None
    
    @staticmethod
    def validate_signup_data(username, email, password, password_confirm):
        """Validate complete signup data"""
        errors = []
        
        # Validate username
        is_valid, error = Validator.validate_username(username)
        if not is_valid:
            errors.append(error)
        
        # Validate email
        is_valid, error = Validator.validate_email(email)
        if not is_valid:
            errors.append(error)
        
        # Validate password
        is_valid, error = Validator.validate_password(password)
        if not is_valid:
            errors.append(error)
        
        # Verify password confirmation
        if password != password_confirm:
            errors.append("Passwords do not match")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_login_data(username, password):
        """Validate login data"""
        errors = []
        
        if not username:
            errors.append("Username is required")
        
        if not password:
            errors.append("Password is required")
        
        return len(errors) == 0, errors


class AuthenticationManager:
    """High-level authentication management"""
    
    def __init__(self, db):
        """Initialize with database instance"""
        self.db = db
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager()
        self.validator = Validator()
    
    def signup(self, username, email, password, password_confirm):
        """
        Register new user
        Returns: (success, data/errors)
        """
        # Validate input
        is_valid, errors = self.validator.validate_signup_data(
            username, email, password, password_confirm
        )
        
        if not is_valid:
            return False, errors
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Create user
        result = self.db.create_user(username, email, password_hash)
        
        if 'error' in result:
            return False, [result['error']]
        
        return True, {'user_id': result['id'], 'username': result['username']}
    
    def login(self, username, password):
        """
        Authenticate user
        Returns: (success, data/errors)
        """
        # Validate input
        is_valid, errors = self.validator.validate_login_data(username, password)
        
        if not is_valid:
            return False, errors
        
        # Get user
        user = self.db.get_user_by_username(username)
        
        if not user:
            return False, ["Username or password is incorrect"]
        
        # Check if user is active
        if not user['is_active']:
            return False, ["Account is deactivated"]
        
        # Verify password
        if not self.password_manager.verify_password(password, user['password']):
            return False, ["Username or password is incorrect"]
        
        # Generate session token
        session_token = self.token_manager.generate_session_token()
        expires_at = self.token_manager.get_token_expiry(hours=24)
        
        # Create session
        session_result = self.db.create_session(
            user['id'], 
            session_token, 
            expires_at=expires_at
        )
        
        if 'error' in session_result:
            return False, [session_result['error']]
        
        return True, {
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'session_token': session_token
        }
    
    def logout(self, session_token):
        """
        Logout user
        Returns: (success, data/errors)
        """
        result = self.db.delete_session(session_token)
        
        if 'error' in result:
            return False, [result['error']]
        
        return True, {'message': 'Logged out successfully'}
    
    def verify_session(self, session_token):
        """
        Verify session token
        Returns: user data if valid, None if invalid
        """
        session = self.db.get_session(session_token)
        
        if not session:
            return None
        
        # Get user
        user = self.db.get_user_by_id(session['user_id'])
        
        if not user or not user['is_active']:
            self.db.delete_session(session_token)
            return None
        
        return {
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email']
        }


# Example usage
if __name__ == '__main__':
    from database import Database
    
    # Initialize database
    db = Database()
    
    # Initialize auth manager
    auth = AuthenticationManager(db)
    
    # Signup
    success, data = auth.signup('john_doe', 'john@example.com', 'Password123', 'Password123')
    print(f"Signup: {success}, {data}")
    
    # Login
    success, data = auth.login('john_doe', 'Password123')
    print(f"Login: {success}, {data}")
    
    # Verify session
    if success:
        user = auth.verify_session(data['session_token'])
        print(f"Session valid: {user}")
