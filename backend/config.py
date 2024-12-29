import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration."""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    DEBUG = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Socket.IO
    SOCKETIO_CORS_ALLOWED_ORIGINS = (
        os.getenv('SOCKETIO_CORS_ORIGIN', '*').split(',')
    )

    # File Upload
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    @staticmethod
    def init_app(app):
        """Initialize application configuration."""
        # Create upload folder if it doesn't exist
        os.makedirs(
            os.path.join(app.root_path, Config.UPLOAD_FOLDER),
            exist_ok=True
        )


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_ECHO = False

    @staticmethod
    def init_app(app):
        """Production-specific configuration."""
        Config.init_app(app)
        # Set production-specific settings here
        app.config['SOCKETIO_CORS_ALLOWED_ORIGINS'] = [
            origin.strip() for origin in 
            os.getenv('SOCKETIO_CORS_ORIGIN', '').split(',')
            if origin.strip()
        ]


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
