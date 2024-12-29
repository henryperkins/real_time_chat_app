"""Authentication middleware."""
import functools
from flask import jsonify, session
from flask_socketio import disconnect


def login_required(f):
    """Decorator to require authentication for routes."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function


def socket_auth_required(f):
    """Decorator to require authentication for WebSocket events."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            disconnect()
            return False
        return f(*args, **kwargs)
    return decorated_function
