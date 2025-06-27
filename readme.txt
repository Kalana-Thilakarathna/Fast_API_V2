# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


#Running the Server

# Option 1: Using the startup script
python run_server.py

# Option 2: Direct uvicorn command
uvicorn main:app --reload --host 127.0.0.1 --port 8000