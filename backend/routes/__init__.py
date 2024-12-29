"""Routes package."""
from .auth import auth_bp
from .projects import projects_bp
from .conversations import conversations_bp
from .models import models_bp

def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(models_bp)
