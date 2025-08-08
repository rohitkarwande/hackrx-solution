import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

print("--- CHECKPOINT 1: SCRIPT START ---")

try:
    print("--- CHECKPOINT 2: LOADING ENVIRONMENT VARIABLES ---")
    load_dotenv()
    print("--- CHECKPOINT 3: ENVIRONMENT VARIABLES LOADED ---")

    # Import our RAG processor after loading env vars
    from rag_processor import generate_answers_from_document
    print("--- CHECKPOINT 4: RAG PROCESSOR IMPORTED ---")

    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("--- FATAL ERROR: GOOGLE_API_KEY NOT FOUND IN ENVIRONMENT ---")
    else:
        print("--- CHECKPOINT 5: GOOGLE_API_KEY FOUND ---")
        try:
            print("--- CHECKPOINT 6: CONFIGURING GOOGLE GENAI ---")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            print("--- CHECKPOINT 7: GOOGLE GENAI CONFIGURED SUCCESSFULLY ---")
        except Exception as e:
            print(f"--- FATAL ERROR CONFIGURING GOOGLE GENAI: {e} ---")

    print("--- CHECKPOINT 8: INITIALIZING FASTAPI APP ---")
    app = FastAPI(title="HackRx 6.0 RAG API")
    print("--- CHECKPOINT 9: FASTAPI APP INITIALIZED ---")

    class HackathonRequest(BaseModel):
        documents: str
        questions: list[str]

    class HackathonResponse(BaseModel):
        answers: list[str]

    @app.get("/")
    def read_root():
        """A simple endpoint to check if the server is alive."""
        print("--- INFO: Root '/' endpoint was called ---")
        return {"status": "ok", "message": "Welcome to the HackRx 6.0 API!"}

    @app.get("/test")
    async def run_test():
        """A simple test endpoint that should respond instantly."""
        print("--- INFO: /test endpoint was called successfully! ---")
        return {"message": "Test successful, the server is running!"}

    @app.post("/hackrx/run", response_model=HackathonResponse)
    async def run_logic(request: HackathonRequest):
        print("--- INFO: Received request on /hackrx/run ---")
        try:
            answers = generate_answers_from_document(
                pdf_url=request.documents,
                questions=request.questions
            )
            print("--- INFO: Sending response from /hackrx/run ---")
            return HackathonResponse(answers=answers)
        except Exception as e:
            print(f"--- CRITICAL ERROR during RAG processing: {e} ---")
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

except Exception as e:
    print(f"--- FATAL ERROR AT GLOBAL LEVEL: {e} ---")