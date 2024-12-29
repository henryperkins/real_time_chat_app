"""Model management routes."""
from flask import Blueprint, request, jsonify, session
from azure_openai_config import azure_openai
from middleware.auth import login_required

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Create blueprint
models_bp = Blueprint('models', __name__, url_prefix='/models')

@models_bp.route('/', methods=['GET'])
@login_required
def list_models():
    """List all available models."""
    try:
        deployments = azure_openai.list_deployments()
        return jsonify(deployments), 200
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to list models"}), 500

@models_bp.route('/active', methods=['POST'])
@login_required
def set_active_model():
    """Set the active model."""
    try:
        data = request.json
        purpose = data.get('purpose')
        if purpose:
            deployment = azure_openai.get_deployment(purpose)
            return jsonify({
                "message": "Active model set successfully",
                "model": deployment.model,
                "purpose": deployment.purpose
            }), 200
        else:
            return jsonify({"error": "Purpose not provided"}), 400
    except Exception as e:
        logger.error(f"Error setting active model: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to set active model"}), 500
