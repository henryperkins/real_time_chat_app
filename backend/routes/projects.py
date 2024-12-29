"""Project management routes."""
import os
import logging
from flask import Blueprint, request, jsonify, session, current_app
from models import db, Project, ProjectFile, Conversation, ConversationParticipant
from middleware.auth import login_required

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
projects_bp = Blueprint('projects', __name__, url_prefix='/projects')


@projects_bp.route('/', methods=['GET', 'POST'])
@login_required
def handle_projects():
    """Handle project listing and creation."""
    try:
        if request.method == 'POST':
            data = request.json
            user_id = session['user_id']
            
            project = Project(
                name=data.get('name'),
                language_model=data.get('language_model', 'gpt-3.5-turbo'),
                creator_id=user_id
            )
            db.session.add(project)
            db.session.commit()
            
            # Create initial conversation for the project
            conversation = Conversation(
                project_id=project.id,
                creator_id=user_id
            )
            db.session.add(conversation)
            db.session.flush()  # Get conversation ID
            
            # Add creator as participant
            participant = ConversationParticipant(
                conversation=conversation,
                user_id=user_id
            )
            db.session.add(participant)
            db.session.commit()
            
            return jsonify({
                "message": "Project created successfully",
                "project_id": project.id,
                "conversation_id": conversation.id
            }), 201
        else:
            user_id = session['user_id']
            # Get projects where user is creator or participant
            projects = Project.query.join(Conversation)\
                .join(ConversationParticipant)\
                .filter(
                    (Project.creator_id == user_id) |
                    (ConversationParticipant.user_id == user_id)
                ).distinct().all()
            
            return jsonify([{
                "id": p.id,
                "name": p.name,
                "language_model": p.language_model,
                "creator_id": p.creator_id,
                "created_at": p.created_at.isoformat()
            } for p in projects]), 200
    except Exception as e:
        logger.error(f"Error in handle_projects: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process project request"}), 500


@projects_bp.route('/<int:project_id>/files', methods=['GET', 'POST'])
@login_required
def handle_project_files(project_id):
    """Handle project file operations."""
    try:
        project = Project.query.get_or_404(project_id)
        user_id = session['user_id']
        
        # Verify user has access to project
        if not (project.creator_id == user_id or 
                ConversationParticipant.query.join(Conversation)
                .filter(
                    Conversation.project_id == project_id,
                    ConversationParticipant.user_id == user_id
                ).first()):
            return jsonify({"error": "Unauthorized"}), 403
        
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
                
            filename = file.filename
            filepath = os.path.join(
                current_app.config['UPLOAD_FOLDER'],
                str(project_id),
                filename
            )
            
            # Create project directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            
            project_file = ProjectFile(
                filename=filename,
                filepath=filepath,
                project_id=project_id
            )
            db.session.add(project_file)
            db.session.commit()
            
            return jsonify({
                "message": "File uploaded successfully",
                "file": {
                    "id": project_file.id,
                    "filename": filename,
                    "uploaded_at": project_file.uploaded_at.isoformat()
                }
            }), 201
        else:
            files = ProjectFile.query.filter_by(project_id=project_id).all()
            return jsonify([{
                "id": f.id,
                "filename": f.filename,
                "uploaded_at": f.uploaded_at.isoformat()
            } for f in files]), 200
    except Exception as e:
        logger.error(f"Error in handle_project_files: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process file request"}), 500
