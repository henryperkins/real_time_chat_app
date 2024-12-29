"""WebSocket event handlers."""
import logging
import asyncio
from datetime import datetime
from flask import session, request
from flask_socketio import join_room, emit
from models import db, Message, User, ConversationParticipant
from middleware.auth import socket_auth_required
from ai_assistant import AIAssistant

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
        def handle_join(data):
            """Handle user joining a conversation room."""
            logger.info(f"handle_join triggered with data: {data}")
            try:
                room = data.get('conversation_id')
                user_id = session['user_id']
                
                # Verify user is a participant
                participant = ConversationParticipant.query.filter_by(
                    conversation_id=room,
                    user_id=user_id
                ).first()
                
                if not participant:
                    logger.warning(
                        f"User {user_id} is not a participant in room {room}"
                        f"Unauthorized join: User {user_id} for room {room}"
                    )
                    emit('error', {'message': 'Failed to join conversation'}, to=request.sid)
                    return False
                
                join_room(room)
                logger.info(f"User {user_id} joined room {room}")
                
                # Get recent messages
                logger.info(f"Fetching recent messages for room {room}")
                messages = Message.query.filter_by(conversation_id=room)\
                    .order_by(Message.created_at.desc())\
                    .limit(50).all()
                
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
                logger.info(f"Emitted recent_messages and status for room {room}")
            except Exception as e:
                logger.error(f"Error in handle_join: {str(e)}", exc_info=True)
                emit('error', {'message': 'Failed to join conversation'}, to=request.sid)
                return False

        @self.socketio.on('send_message')
        @socket_auth_required
        def handle_send_message(data):
            """Handle sending a message in a conversation."""
            logger.info(f"handle_send_message triggered with data: {data}")
            try:
                room = data.get('conversation_id')
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
                        f"Unauthorized message: User {user_id} for room {room}"
                    )
                    emit('error', {'message': 'Failed to send message'}, to=request.sid)
                    return False
                
                # Save message to database
                logger.info(f"Attempting to save user message: {message_text}")
                new_message = Message(
                    content=message_text,
                    conversation_id=room,
                    user_id=user_id
                )
                db.session.add(new_message)
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Error saving user message: {str(e)}", exc_info=True)
                    emit('error', {'message': 'Failed to send message'}, to=request.sid)
                    return False
                
                # Get user information
                user = User.query.get(user_id)
                
                # Generate AI response
                conversation_history = Message.query.filter_by(
                    conversation_id=room
                ).order_by(Message.created_at.desc()).limit(10).all()
                
                logger.info(f"Generating AI response for message: {message_text}")
                async def generate_and_emit_ai_response():
                    try:
                        ai_response = self.ai_assistant.get_ai_response(
                            message_text,
                            [msg.content for msg in conversation_history],
                            project_id
                        )
                        logger.info(f"AI response generated: {ai_response}")
                        
                        # Emit user message to room
                        emit(
                            "message",
                            {
                                "message": message_text,
                                "user_id": user_id,
                                "username": user.username,
                                "project_id": project_id,
                                "created_at": new_message.created_at.isoformat()
                            },
                            to=room
                        )
                        
                        # Emit AI response to room
                        emit(
                            "message",
                            {
                                "message": ai_response,
                                "user_id": None,
                                "username": "AI Assistant",
                                "project_id": project_id,
                                "created_at": datetime.utcnow().isoformat()
                            },
                            to=room
                        )
                        logger.info(f"Emitted AI response to room {room}")
                        
                        # Save AI response
                        logger.info(f"Attempting to save AI message: {ai_response}")
                        ai_message = Message(
                            content=ai_response,
                            conversation_id=room,
                            user_id=None,  # AI user
                            ai_response=True
                        )
                        db.session.add(ai_message)
                        try:
                            db.session.commit()
                        except Exception as e:
                            logger.error(f"Error saving AI message: {str(e)}", exc_info=True)
                            emit('error', {'message': 'Failed to save AI response'}, to=request.sid)
                    except Exception as e:
                        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
                        emit('error', {'message': f'Failed to generate AI response: {str(e)}'}, to=request.sid)
                
                asyncio.create_task(generate_and_emit_ai_response())
            except Exception as e:
                logger.error(f"Error in handle_send_message: {str(e)}", exc_info=True)
                emit('error', {'message': f'Failed to process message: {str(e)}'}, to=request.sid)
                return False
