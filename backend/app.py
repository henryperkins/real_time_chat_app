from sqlalchemy.exc import OperationalError
import os
import logging
from datetime import timedelta
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import asyncio
from dotenv import load_dotenv

from config import config
from models import db
from routes import register_blueprints
from sockets.handlers import WebSocketManager
from azure_openai_config import AzureOpenAIConfig

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def create_azure_openai_config():
    """Create and initialize AzureOpenAIConfig asynchronously."""
    global azure_openai
    try:
        # Ensure environment variables are loaded
        load_dotenv(dotenv_path='../.env')
        azure_openai = AzureOpenAIConfig()
        await azure_openai.async_init()
    except Exception as e:
        logger.error(
            f"Failed to initialize AzureOpenAIConfig: {str(e)}. "
            f"Check AZURE_OPENAI_ENDPOINT and network connectivity.",
            exc_info=True
        )
        azure_openai = None  # Ensure it is explicitly set to None

def create_app(config_name='development'):
    """Create and configure the Flask application."""
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_folder = os.path.join(base_dir, 'frontend', 'public')
    uploads_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    # Ensure directories exist
    os.makedirs(uploads_folder, exist_ok=True)
    logger.info(f"Static folder path: {static_folder}")
    logger.info(f"Uploads folder path: {uploads_folder}")
    
    # Create Flask app
    app = Flask(__name__, static_folder=static_folder, static_url_path='')
    app.config['UPLOAD_FOLDER'] = uploads_folder
    
    # Enable detailed error logging
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Set session lifetime
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
    
    # Initialize extensions
    db.init_app(app)
    # Initialize database
    with app.app_context():
        db.create_all()
    CORS(app,
         supports_credentials=True,
         resources={
             r"/*": {
                 "origins": app.config.get('SOCKETIO_CORS_ALLOWED_ORIGINS', '*'),
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type"],
                 "expose_headers": ["Content-Range", "X-Content-Range"],
                 "supports_credentials": True
             }
         })
    
    # Register blueprints
    register_blueprints(app)
    
    # Serve frontend files
    @app.route('/')
    def index():
        try:
            logger.info(f"Serving login.html from {app.static_folder}")
            return app.send_static_file('login.html')
        except Exception as e:
            logger.error(f"Failed to serve login.html: {str(e)}", exc_info=True)
            raise

    @app.route('/favicon.ico')
    def favicon():
        return app.send_static_file('favicon.ico')

    @app.route('/<path:filename>')
    def serve_static(filename):
        try:
            logger.info(f"Serving static file: {filename}")
            response = app.send_static_file(filename)
            response.headers['Cache-Control'] = 'public, max-age=300'  # 5 minutes
            return response
        except Exception as e:
            logger.error(f"Failed to serve {filename}: {str(e)}", exc_info=True)
            return '', 404
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        logger.error(f"404 error: {str(error)}", exc_info=True)
        return jsonify({
            "error": "Resource not found",
            "details": str(error),
            "status_code": 404
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {str(error)}", exc_info=True)
        return jsonify({
            "error": "An internal server error occurred",
            "details": str(error),
            "status_code": 500
        }), 500

    @app.errorhandler(Exception)
    def handle_error(error):
        logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(error),
            "status_code": 500
        }), 500
    
    @app.errorhandler(OperationalError)
    def handle_db_error(error):
        logger.error(f"Database error: {str(error)}", exc_info=True)
        return jsonify({
            "error": "A database error occurred",
            "details": str(error),
            "status_code": 500
        }), 500
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Initialize AzureOpenAIConfig asynchronously
    asyncio.run(create_azure_openai_config())

    return app

def create_socketio(app):
    """Create and configure SocketIO for the application."""
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        logger=True,
        engineio_logger=True,
        async_mode='threading'
    )
    
    # Initialize WebSocket handlers
    WebSocketManager(socketio)
    
    return socketio

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    socketio = create_socketio(app)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
