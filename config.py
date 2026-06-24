"""
CIFAKE Configuration Module

Application settings and constants for the CIFAKE project.
"""

import os
from pathlib import Path

# ============= PROJECT PATHS =============
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / 'dataset'
MODELS_DIR = PROJECT_ROOT / 'models'
STATIC_DIR = PROJECT_ROOT / 'static'
TEMPLATES_DIR = PROJECT_ROOT / 'templates'
UPLOADS_DIR = STATIC_DIR / 'uploads'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

# ============= FLASK CONFIGURATION =============
DEBUG = os.getenv('FLASK_DEBUG', True)
TESTING = False
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# ============= FILE UPLOAD CONFIGURATION =============
UPLOAD_FOLDER = str(UPLOADS_DIR)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size

# ============= DATABASE CONFIGURATION =============
DATABASE_PATH = os.getenv('DATABASE_PATH', PROJECT_ROOT / 'predictions.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# ============= MODEL CONFIGURATION =============
MODEL_PATH = os.getenv('MODEL_PATH', MODELS_DIR / 'best_model.keras')
MODEL_INPUT_SIZE = 32  # CIFAR-10 image size
MODEL_NUM_CLASSES = 2  # Binary: Real (0) or Fake (1)
MODEL_CONFIDENCE_THRESHOLD = 0.5

# ============= IMAGE PROCESSING =============
IMAGE_SIZE = (32, 32)  # Input size for model
NORMALIZE_MEAN = [0.5, 0.5, 0.5]  # Normalization parameters
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ============= DATA SPLIT CONFIGURATION =============
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============= TRAINING CONFIGURATION =============
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2

# ============= CLASS LABELS =============
CLASS_LABELS = {
    0: 'REAL',
    1: 'FAKE'
}
CLASS_LABELS_REVERSE = {
    'REAL': 0,
    'FAKE': 1
}

# ============= PREDICTION CONFIDENCE =============
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,
    'medium': 0.70,
    'low': 0.50
}

# ============= GRAD-CAM CONFIGURATION =============
GRADCAM_LAYER = 'conv2d_3'  # Layer to visualize (adjust based on model)
GRADCAM_HEATMAP_ALPHA = 0.4  # Transparency of overlay

# ============= LOGGING CONFIGURATION =============
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = PROJECT_ROOT / 'app.log'

# ============= PERFORMANCE TARGETS =============
TARGET_ACCURACY = 0.9298  # 92.98%
TARGET_FALSE_POSITIVE_RATE = 0.05  # < 5%
TARGET_FALSE_NEGATIVE_RATE = 0.05  # < 5%
TARGET_INFERENCE_TIME = 0.5  # < 500ms

# ============= ENVIRONMENT =============
ENVIRONMENT = os.getenv('FLASK_ENV', 'development')
DEBUG_MODE = ENVIRONMENT == 'development'

# ============= API CONFIGURATION =============
JSON_SORT_KEYS = False
JSONIFY_PRETTYPRINT_REGULAR = DEBUG_MODE

# ============= SESSION CONFIGURATION =============
PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
SESSION_COOKIE_SECURE = not DEBUG_MODE
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# ============= CORS CONFIGURATION =============
CORS_ORIGINS = ['*'] if DEBUG_MODE else ['localhost', '127.0.0.1']

# ============= FLASK CONFIG CLASS =============
class Config:
    """Flask configuration class"""
    
    # Flask settings
    DEBUG = os.getenv('FLASK_DEBUG', False)
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 8000))
    
    # File upload
    UPLOAD_FOLDER = str(UPLOADS_DIR)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', str(PROJECT_ROOT / 'database.db'))
    DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
    
    # Session
    PERMANENT_SESSION_LIFETIME = 86400
    SESSION_COOKIE_SECURE = not DEBUG
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # JSON
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = DEBUG
    
    # CORS
    CORS_ORIGINS = ['*'] if DEBUG else ['localhost']
    
    # Directories
    PROJECT_ROOT = PROJECT_ROOT
    DATASET_DIR = DATASET_DIR
    MODELS_DIR = MODELS_DIR
    STATIC_DIR = STATIC_DIR
    TEMPLATES_DIR = TEMPLATES_DIR
    UPLOADS_DIR = UPLOADS_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', str(MODELS_DIR / 'best_model.hdf5'))
    MODEL_INPUT_SIZE = 32
    MODEL_NUM_CLASSES = 2
    
    # Image settings
    IMAGE_SIZE = (32, 32)
    NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    NORMALIZE_STD = [0.5, 0.5, 0.5]


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False


# Config selector
config_selector = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}


# ============= CREATE REQUIRED DIRECTORIES =============
def init_directories():
    """Create required directories if they don't exist."""
    directories = [
        DATASET_DIR,
        MODELS_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on module import
init_directories()
