from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    language_model = db.Column(
        db.String(50),
        default='gpt-3.5-turbo'
    )
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    creator_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    
    # Relationships
    files = db.relationship(
        'ProjectFile',
        backref='project',
        lazy=True
    )
    conversations = db.relationship(
        'Conversation',
        backref='project',
        lazy=True
    )


class ProjectFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(
        db.Integer,
        db.ForeignKey('project.id'),
        nullable=False
    )


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Nullable for independent conversations
    project_id = db.Column(
        db.Integer,
        db.ForeignKey('project.id'),
        nullable=True
    )
    creator_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    
    # Relationships
    messages = db.relationship(
        'Message',
        backref='conversation',
        lazy=True
    )
    participants = db.relationship(
        'ConversationParticipant',
        backref='conversation',
        lazy=True
    )


class ConversationParticipant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey('conversation.id'),
        nullable=False
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
        nullable=False
    )
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (
        db.Index('idx_conversation_user', 'conversation_id', 'user_id'),
    )


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    conversation_id = db.Column(
        db.Integer,
        db.ForeignKey('conversation.id'),
        nullable=False
    )
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sentiment_score = db.Column(db.Float, nullable=True)
    ai_response = db.Column(db.Text, nullable=True)
