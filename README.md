# Real-Time Chat Application Setup

## Environment Variables Setup

To run this application, you need to set up the following environment variables:

1. **Using a `.env` file**:
   - Create a `.env` file in the root directory of your project.
   - Add the following lines to the `.env` file:
     ```
     AZURE_OPENAI_ENDPOINT=your_endpoint_here
     AZURE_OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_endpoint_here` and `your_api_key_here` with your actual Azure OpenAI endpoint and API key.

2. **Setting environment variables directly in the terminal**:
   - Before running your application, set the environment variables in your terminal:
     - On Unix/Linux/Mac:
       ```bash
       export AZURE_OPENAI_ENDPOINT=your_endpoint_here
       export AZURE_OPENAI_API_KEY=your_api_key_here
       ```
     - On Windows (Command Prompt):
       ```cmd
       set AZURE_OPENAI_ENDPOINT=your_endpoint_here
       set AZURE_OPENAI_API_KEY=your_api_key_here
       ```
     - On Windows (PowerShell):
       ```powershell
       $env:AZURE_OPENAI_ENDPOINT='your_endpoint_here'
       $env:AZURE_OPENAI_API_KEY='your_api_key_here'
       ```

## Running the Application

After setting the environment variables, run your application:

