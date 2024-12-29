from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from typing import Optional, List, Any
from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.orm.relationships import RelationshipProperty


db = SQLAlchemy()


class User(db.Model):
    """User model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    username: str = db.Column(db.String(80), unique=True, nullable=False)
    password_hash: str = db.Column(db.String(256), nullable=False)
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, username: str, password_hash: str) -> None:
        """Initialize a new user."""
        self.username = username
        self.password_hash = password_hash


class Project(db.Model):
    """Project model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String(100), nullable=False)
    language_model: str = db.Column(
        db.String(50),
        default='gpt-3.5-turbo'
    )
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)
    creator_id: int = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    
    # Relationships
    files: Mapped[List['ProjectFile']] = relationship(
        'ProjectFile',
        backref='project',
        lazy=True
    )
    conversations: Mapped[List['Conversation']] = relationship(
        'Conversation',
        backref='project',
        lazy=True
    )


class ProjectFile(db.Model):
    """Project file model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    filename: str = db.Column(db.String(255), nullable=False)
    filepath: str = db.Column(db.String(512), nullable=False)
    uploaded_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)
    project_id: int = db.Column(
        db.Integer,
        db.ForeignKey('project.id'),
        nullable=False
    )


class Conversation(db.Model):
    """Conversation model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)
    project_id: Optional[int] = db.Column(
        db.Integer,
        db.ForeignKey('project.id'),
        nullable=True
    )
    creator_id: int = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    
    # Relationships
    messages: Mapped[List['Message']] = relationship(
        'Message',
        backref='conversation',
        lazy=True
    )
    participants: Mapped[List['ConversationParticipant']] = relationship(
        'ConversationParticipant',
        backref='conversation',
        lazy=True
    )


class ConversationParticipant(db.Model):
    """Conversation participant model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    conversation_id: int = db.Column(
        db.Integer,
        db.ForeignKey('conversation.id'),
        nullable=False
    )
    user_id: int = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
<<<<<<< HEAD
    joined_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, conversation_id: int, user_id: int) -> None:
        """Initialize a new conversation participant."""
        self.conversation_id = conversation_id
        self.user_id = user_id
=======
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (
        db.Index('idx_conversation_user', 'conversation_id', 'user_id'),
    )
>>>>>>> 3f131548aae059f728b4edd5d7dc3636158ff180


class Message(db.Model):
    """Message model with type hints."""
    id: int = db.Column(db.Integer, primary_key=True)
    content: str = db.Column(db.Text, nullable=False)
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow)
    conversation_id: int = db.Column(
        db.Integer,
        db.ForeignKey('conversation.id'),
        nullable=False
    )
    user_id: int = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sentiment_score: Optional[float] = db.Column(db.Float, nullable=True)
    ai_response: Optional[str] = db.Column(db.Text, nullable=True)

    def __init__(
        self,
        content: str,
        conversation_id: int,
        user_id: int,
        ai_response: Optional[str] = None,
        sentiment_score: Optional[float] = None
    ) -> None:
        """Initialize a new message."""
        self.content = content
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.ai_response = ai_response
        self.sentiment_score = sentiment_score
