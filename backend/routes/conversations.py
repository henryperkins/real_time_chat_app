"""Conversation management routes."""
import logging
from flask import Blueprint, request, jsonify, session
from models import (
    db, Conversation, ConversationParticipant,
    Message, User
)
from middleware.auth import login_required

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint
conversations_bp = Blueprint('conversations', __name__, url_prefix='/conversations')


@conversations_bp.route('/', methods=['GET', 'POST'])
@login_required
def handle_conversations():
    """Handle conversation listing and creation."""
    try:
        user_id = session['user_id']
        
        if request.method == 'POST':
            data = request.json
            project_id = data.get('project_id')  # Optional
            participant_ids = data.get('participant_ids', [])
            
            # Ensure creator is in participant list
            if user_id not in participant_ids:
                participant_ids.append(user_id)
            
            conversation = Conversation(
                creator_id=user_id,
                project_id=project_id
            )
            db.session.add(conversation)
            db.session.flush()
            
            # Add participants
            for participant_id in participant_ids:
                participant = ConversationParticipant(
                    conversation_id=conversation.id,
                    user_id=participant_id
                )
                db.session.add(participant)
            
            db.session.commit()
            
            return jsonify({
                "message": "Conversation created successfully",
                "conversation_id": conversation.id,
                "project_id": project_id
            }), 201
        else:
            # Get conversations where user is a participant
            conversations = Conversation.query\
                .join(ConversationParticipant)\
                .filter(ConversationParticipant.user_id == user_id)\
                .all()
            
            return jsonify([{
                "id": c.id,
                "project_id": c.project_id,
                "creator_id": c.creator_id,
                "created_at": c.created_at.isoformat(),
                "participants": [{
                    "user_id": p.user_id,
                    "joined_at": p.joined_at.isoformat()
                } for p in c.participants]
            } for c in conversations]), 200
    except Exception as e:
        logger.error(f"Error in handle_conversations: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process conversation request"}), 500


@conversations_bp.route('/<int:conversation_id>/messages', methods=['GET', 'POST'])
@login_required
def handle_messages(conversation_id):
    """Handle message operations for a conversation."""
    try:
        user_id = session['user_id']
        
        # Verify user is a participant
        participant = ConversationParticipant.query.filter_by(
            conversation_id=conversation_id,
            user_id=user_id
        ).first()
        
        if not participant:
            return jsonify({"error": "Unauthorized"}), 403
        
        if request.method == 'POST':
            data = request.json
            content = data.get('content')
            
            message = Message(
                content=content,
                conversation_id=conversation_id,
                user_id=user_id
            )
            db.session.add(message)
            db.session.commit()
            
            # Get user information
            user = User.query.get(user_id)
            
            return jsonify({
                "message": "Message sent successfully",
                "message_id": message.id,
                "content": content,
                "user_id": user_id,
                "username": user.username,
                "created_at": message.created_at.isoformat()
            }), 201
        else:
            # Get messages with pagination
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            
            messages = Message.query\
                .filter_by(conversation_id=conversation_id)\
                .order_by(Message.created_at.desc())\
                .paginate(page=page, per_page=per_page, error_out=False)
            
            return jsonify({
                "messages": [{
                    "id": m.id,
                    "content": m.content,
                    "user_id": m.user_id,
                    "created_at": m.created_at.isoformat()
                } for m in messages.items],
                "total": messages.total,
                "pages": messages.pages,
                "current_page": messages.page
            }), 200
    except Exception as e:
        logger.error(f"Error in handle_messages: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process message request"}), 500


@conversations_bp.route('/users', methods=['GET'])
@login_required
def get_users():
    """Get users for adding to conversations."""
    try:
        search = request.args.get('search', '')
        users = User.query\
            .filter(User.username.ilike(f'%{search}%'))\
            .limit(10)\
            .all()
        
        return jsonify([{
            "id": u.id,
            "username": u.username
        } for u in users]), 200
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch users"}), 500
