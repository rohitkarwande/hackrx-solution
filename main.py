import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# This is our "brain" file we created in the last step
from rag_processor import generate_answers_from_document

# This function will load our secret API key from a file
# We will create this file in the next step
print("INFO: Loading environment variables...")
load_dotenv()

# We need to configure the genai library with our key
# It's important to do this here in the main file when the app starts
import google.generativeai as genai
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not found.")
else:
    print("INFO: Google API Key loaded successfully.")
    genai.configure(api_key=api_key)


# Initialize our FastAPI application
app = FastAPI(title="HackRx 6.0 RAG API")

# Define the structure of the incoming request JSON
class HackathonRequest(BaseModel):
    documents: str
    questions: list[str]

# Define the structure of the outgoing response JSON
class HackathonResponse(BaseModel):
    answers: list[str]



@app.get("/", summary="Check if the API is running")
def read_root():
    """A simple endpoint to check if the server is alive."""
    return {"status": "ok", "message": "Welcome to the HackRx 6.0 API!"}

@app.get("/test")
async def run_test():
    """A simple test endpoint that should respond instantly."""
    print("INFO: /test endpoint was called successfully!")
    return {"message": "Test successful, the server is running!"}


@app.post("/hackrx/run", response_model=HackathonResponse, summary="Process a document and questions")
async def run_logic(request: HackathonRequest):
    """
    This is the main endpoint the hackathon platform will call.
    It takes a document URL and a list of questions, and returns answers.
    """
    print("INFO: Received request on /hackrx/run")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server is not configured with a Google API Key.")

    try:
        # Here we call our "brain" function from the other file
        print("INFO: Handing off request to the RAG processor...")
        answers = generate_answers_from_document(
            pdf_url=request.documents,
            questions=request.questions
        )
        print("INFO: RAG processor returned answers. Sending response.")
        return HackathonResponse(answers=answers)
    
    except Exception as e:
        # If anything goes wrong in our "brain", we catch the error
        print(f"CRITICAL ERROR during RAG processing: {e}")
        # And return a helpful error message to the user
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")