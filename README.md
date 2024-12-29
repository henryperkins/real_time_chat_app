# Real-Time Chat Application

A modular Flask-based real-time chat application that supports project-based and independent conversations.

## Project Structure

```
backend/
├── routes/              # Route handlers
│   ├── auth.py         # Authentication routes
│   ├── projects.py     # Project management routes
│   └── conversations.py # Conversation routes
├── sockets/            # WebSocket handlers
│   └── handlers.py     # WebSocket event handlers
├── middleware/         # Middleware components
│   └── auth.py        # Authentication middleware
├── utils/             # Utility functions
├── models.py          # Database models
├── config.py          # Application configuration
└── app.py            # Main application entry point

frontend/
├── index.html         # Main chat interface
└── project_management.html # Project management interface
```

## Features

- Real-time messaging using WebSocket
- Project-based conversations with file management
- Independent conversations
- User authentication and session management
- Secure WebSocket communication
- File upload support for projects

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up environment variables in `backend/.env`:
```
SECRET_KEY=your_secret_key_here
DEBUG_MODE=true
DATABASE_URL=sqlite:///real_time_chat.db
SOCKETIO_CORS_ORIGIN=*
UPLOAD_FOLDER=uploads
```

3. Initialize the database:
```bash
cd backend
flask db init
flask db migrate
flask db upgrade
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Open the frontend in a web browser:
- Navigate to `frontend/index.html` for the chat interface
- Navigate to `frontend/project_management.html` for project management

## API Endpoints

### Authentication
- `POST /auth/register` - Register a new user
- `POST /auth/login` - Login user
- `POST /auth/logout` - Logout user
- `GET /auth/check` - Check authentication status

### Projects
- `GET /projects` - List all accessible projects
- `POST /projects` - Create a new project
- `GET /projects/<id>/files` - List project files
- `POST /projects/<id>/files` - Upload file to project

### Conversations
- `GET /conversations` - List user's conversations
- `POST /conversations` - Create a new conversation
- `GET /conversations/<id>/messages` - Get conversation messages
- `POST /conversations/<id>/messages` - Send a message
- `GET /conversations/users` - Search users for adding to conversations

## WebSocket Events

### Client -> Server
- `join` - Join a conversation room
- `send_message` - Send a message to a conversation

### Server -> Client
- `message` - New message in conversation
- `status` - Room status updates
- `recent_messages` - Recent messages when joining a room

## Security Features

- Session-based authentication
- WebSocket authentication middleware
- CORS protection
- File upload security
- SQL injection protection through SQLAlchemy
- Password hashing with Werkzeug

## Error Handling

The application includes comprehensive error handling:
- Input validation
- Authentication checks
- File upload validation
- Database error handling
- WebSocket error handling
- Logging for debugging

## Development

The application follows a modular structure for maintainability:
- Route handlers are organized into blueprints
- WebSocket events are managed through a dedicated manager
- Authentication is handled through middleware
- Database models are clearly defined
- Configuration is environment-based
