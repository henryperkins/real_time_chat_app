"""WebSocket event handlers."""

import logging
from typing import Optional, Dict, Any, cast
from flask import session, request
from flask_socketio import join_room, emit
from sqlalchemy.orm import Session
from sqlalchemy import text
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

        @self.socketio.on("join")
        @socket_auth_required
        async def handle_join(data):
            """Handle user joining a conversation room."""
            logger.info(f"handle_join triggered with data: {data}")
            try:
                room = data.get("conversation_id")
                if not room:
                    logger.error("No conversation_id provided")
                    emit(
                        "error",
                        {"message": "No conversation_id provided"},
                        to=request.sid,
                    )
                    return False

                user_id = session.get("user_id")
                participant = ConversationParticipant.query.filter_by(
                    conversation_id=room, user_id=user_id
                ).first()

                if not participant:
                    logger.warning(f"Unauthorized join: User {user_id} for room {room}")
                    emit("error", {"message": "Unauthorized access"}, to=request.sid)
                    return False

                join_room(room)
                logger.info(f"User {user_id} joined room {room}")

                messages = (
                    Message.query.filter_by(conversation_id=room)
                    .order_by(Message.created_at.desc())
                    .limit(50)
                    .all()
                )

                emit(
                    "recent_messages",
                    {
                        "messages": [
                            {
                                "content": msg.content,
                                "user_id": msg.user_id,
                                "created_at": msg.created_at.isoformat(),
                            }
                            for msg in reversed(messages)
                        ]
                    },
                    to=request.sid,
                )

                emit(
                    "status",
                    {"message": f"User joined room {room}", "user_id": user_id},
                    to=room,
                )

            except Exception as e:
                logger.error(f"Error in handle_join: {str(e)}", exc_info=True)
                emit(
                    "error", {"message": "Failed to join conversation"}, to=request.sid
                )

        @self.socketio.on("send_message")
        @socket_auth_required
        async def handle_send_message(data):
            """Handle sending a message in a conversation."""
            logger.info(f"handle_send_message triggered with data: {data}")
            try:
                room = data.get("conversation_id")
                if not room:
                    logger.error("No conversation_id provided")
                    emit(
                        "error",
                        {"message": "No conversation_id provided"},
                        to=request.sid,
                    )
                    return False

                message_text = data.get("message")
                user_id = session.get("user_id")
                project_id = data.get("project_id")

                participant = ConversationParticipant.query.filter_by(
                    conversation_id=room, user_id=user_id
                ).first()

                if not participant:
                    logger.warning(
                        f"Unauthorized message: User {user_id} for room {room}"
                    )
                    emit("error", {"message": "Unauthorized access"}, to=request.sid)
                    return False

                # Save user message to the database
                new_message = Message(
                    content=message_text, conversation_id=room, user_id=user_id
                )
                db.session.add(new_message)
                db.session.commit()

                # Get user information
                user = User.query.get(user_id)
                if not user:
                    logger.error(f"User {user_id} not found")
                    emit("error", {"message": "User not found"}, to=request.sid)
                    return False

                # Emit user message
                emit(
                    "message",
                    {
                        "message": new_message.content,
                        "user_id": user.id,
                        "username": user.username,
                        "project_id": project_id,
                        "created_at": new_message.created_at.isoformat(),
                        "is_ai": False,
                    },
                    to=room,
                )

                # Notify that AI is typing
                emit(
                    "ai_status",
                    {"status": "typing", "user_id": None, "username": "AI Assistant"},
                    to=room,
                )

                # Generate AI response asynchronously
                history = (
                    Message.query.filter(Message.conversation_id == room)
                    .order_by(text("created_at DESC"))
                    .limit(10)
                    .all()
                )
                ai_response = await self.ai_assistant.get_ai_response(
                    message_text, [msg.content for msg in reversed(history)], project_id
                )

                if ai_response.get("error"):
                    emit(
                        "ai_status",
                        {
                            "status": "error",
                            "message": ai_response["content"],
                            "user_id": None,
                        },
                        to=room,
                    )
                    return

                # Save AI response to the database
                ai_user = User.query.filter_by(username="AI Assistant").first()
                if not ai_user:
                    ai_user = User(
                        username="AI Assistant", password_hash="$ai$assistant$not$used$"
                    )
                    db.session.add(ai_user)
                    db.session.commit()

                ai_message = Message(
                    content=ai_response["content"],
                    conversation_id=room,
                    user_id=ai_user.id,
                )
                db.session.add(ai_message)
                db.session.commit()

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
                        "finish_reason": ai_response.get("finish_reason"),
                    },
                    to=room,
                )

                # Notify that AI is done typing
                emit(
                    "ai_status",
                    {
                        "status": "completed",
                        "user_id": ai_user.id,
                        "username": "AI Assistant",
                    },
                    to=room,
                )

            except Exception as e:
                logger.error(f"Error in handle_send_message: {str(e)}", exc_info=True)
                emit("error", {"message": "Failed to process message"}, to=request.sid)
