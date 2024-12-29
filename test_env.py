import sys
import os

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.append(backend_path)

from app import create_app
app = create_app()
with app.app_context():
    import os
    print('AZURE_OPENAI_ENDPOINT:', os.getenv('AZURE_OPENAI_ENDPOINT'))
    print('AZURE_OPENAI_KEY:', os.getenv('AZURE_OPENAI_KEY'))
    print('AZURE_OPENAI_API_VERSION:', os.getenv('AZURE_OPENAI_API_VERSION'))
    print('AZURE_OPENAI_DEPLOYMENT:', os.getenv('AZURE_OPENAI_DEPLOYMENT'))
