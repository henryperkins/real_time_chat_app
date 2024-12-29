"""WebSocket event handlers."""
import logging
from typing import Optional, Dict, Any, cast
from flask import session
from flask_socketio import join_room, emit
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from models import db, Message, User, ConversationParticipant
from middleware.auth import socket_auth_required
from ai_assistant import AIAssistant

# Type aliases
MessageDict = Dict[str, Any]
UserDict = Dict[str, Any]

# Set up logging
logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket events and handlers."""
    
    def __init__(self, socketio):
        """Initialize the WebSocket manager."""
        self.socketio = socketio
        self.ai_assistant = AIAssistant()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up WebSocket event handlers."""
        @self.socketio.on('join')
        @socket_auth_required
        async def handle_join(data):
            """Handle user joining a conversation room."""
            logger.info("handle_send_message triggered with data: %s", data)
            try:
                room = data.get('conversation_id')
                if not room:
                    logger.error("No conversation_id provided")
                    return False

                user_id = session['user_id']
                
                # Verify user is a participant
                participant = ConversationParticipant.query.filter_by(
                    conversation_id=room,
                    user_id=user_id
                ).first()
                
                if not participant:
                    logger.warning(
                        "Unauthorized join: User %d for room %d",
                        user_id, room
                    )
                    return False
                
                join_room(room)
                logger.info("User %d joined room %d", user_id, room)

                # Get recent messages using text() for ordering
                messages = (
                    Message.query
                    .filter(Message.conversation_id == room)
                    .order_by(text('created_at DESC'))
                    .limit(50)
                    .all()
                )
                
                # Emit recent messages to the user
                emit('recent_messages', {
                    'messages': [{
                        'content': msg.content,
                        'user_id': msg.user_id,
                        'created_at': msg.created_at.isoformat()
                    } for msg in reversed(messages)]
                })
                
                emit(
                    "status",
                    {
                        "message": f"User joined room {room}",
                        "user_id": user_id
                    },
                    to=room
                )
            except Exception as e:
                logger.error("Error in handle_join: %s", str(e), exc_info=True)
                return False

        @self.socketio.on('send_message')
        @socket_auth_required
        async def handle_send_message(data):
            """Handle sending a message in a conversation."""
            room = None
            try:
                room = data.get('conversation_id')
                if not room:
                    logger.error("No conversation_id provided")
                    return False

                message_text = data.get('message')
                user_id = session['user_id']
                project_id = data.get('project_id')
                
                # Verify user is a participant
                participant = ConversationParticipant.query.filter_by(
                    conversation_id=room,
                    user_id=user_id
                ).first()
                
                if not participant:
                    logger.warning(
                        "Unauthorized message: User %d for room %d",
                        user_id, room
                    )
                    return False
                
                # Get user information
                user = User.query.get(user_id)
                if not user:
                    logger.error(f"User {user_id} not found")
                    return False

                # Get or create AI user with proper type checking
                ai_user = cast(Optional[User], User.query.filter_by(username='AI Assistant').first())
                if not ai_user:
                    # Create AI user with a dummy password hash
                    ai_user = User(
                        username='AI Assistant',
                        password_hash='$ai$assistant$not$used$'
                    )
                    db.session.add(ai_user)
                    db.session.commit()
                    # Refresh after commit to ensure all attributes are loaded
                    db.session.refresh(ai_user)

                try:
                    # Save user message to database with proper type hints
                    new_message = Message(
                        content=str(message_text),
                        conversation_id=int(room),
                        user_id=int(user_id),
                        ai_response=None  # Not an AI response
                    )
                    db.session.add(new_message)
                    db.session.commit()
                    # Refresh to ensure all attributes are loaded
                    db.session.refresh(new_message)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid data types: {str(e)}")
                    return False

                # Get recent conversation history using text() for ordering
                history = (
                    Message.query
                    .filter(Message.conversation_id == room)
                    .order_by(text('created_at DESC'))
                    .limit(10)
                    .all()
                )
                
                # Emit user message first
                emit(
                    "message",
                    {
                        "message": new_message.content,
                        "user_id": user.id,
                        "username": user.username,
                        "project_id": project_id,
                        "created_at": new_message.created_at.isoformat(),
                        "is_ai": False
                    },
                    to=room
                )

                # Notify that AI is typing
                emit(
                    "ai_status",
                    {
                        "status": "typing",
                        "user_id": ai_user.id,
                        "username": "AI Assistant"
                    },
                    to=room
                )
                
                # Generate AI response asynchronously
                ai_response = await self.ai_assistant.get_ai_response(
                    message_text,
                    [msg.content for msg in reversed(history)],
                    project_id
                )
                
                # Check for errors in AI response
                if ai_response.get("error"):
                    emit(
                        "ai_status",
                        {
                            "status": "error",
                            "message": ai_response["content"],
                            "user_id": ai_user.id
                        },
                        to=room
                    )
                    return
                
                try:
                    # Save AI response with proper type hints
                    ai_message = Message(
                        content=str(ai_response["content"]),
                        conversation_id=int(room),
                        user_id=int(ai_user.id),
                        ai_response=str(ai_response["content"])  # Store the actual response
                    )
                    db.session.add(ai_message)
                    db.session.commit()
                    # Refresh to ensure all attributes are loaded
                    db.session.refresh(ai_message)
                except (ValueError, TypeError, KeyError) as e:
                    logger.error(f"Error saving AI response: {str(e)}")
                    emit(
                        "error",
                        {
                            "message": "Failed to save AI response",
                            "details": str(e)
                        },
                        to=room
                    )
                    return False

                # Emit AI response
                emit(
                    "message",
                    {
                        "message": ai_message.content,
                        "user_id": ai_user.id,
                        "username": "AI Assistant",
                        "project_id": project_id,
                        "created_at": ai_message.created_at.isoformat(),
                        "is_ai": True,
                        "model": ai_response.get("model"),
                        "finish_reason": ai_response.get("finish_reason")
                    },
                    to=room
                )

                # Notify that AI is done typing
                emit(
                    "ai_status",
                    {
                        "status": "completed",
                        "user_id": ai_user.id,
                        "username": "AI Assistant"
                    },
                    to=room
                )

            except Exception as e:
                logger.error("Error in handle_send_message: %s", str(e))
                if room:  # Only emit if we have a valid room
                    emit(
                        "error",
                        {
                            "message": "Failed to process message",
                            "details": str(e)
                        },
                        to=room
                    )
                return False
