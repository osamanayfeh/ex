import os
from huggingface_hub import login

def hugging_face_auth():
    # getting token from .env
    HUGGING_FACE_TOKEN= os.environ.get("HF_TOKEN")
    
    # Logging in
    print("Attempting Hugging face login...")
    login(token=HUGGING_FACE_TOKEN)
    print("Login successful")
