import os
import io
import pypdf
import requests
import numpy as np
import google.generativeai as genai

# Get the Hugging Face API URL and Token from environment variables
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_text_from_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    print("INFO: Fetching PDF from URL...")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Could not download PDF. {e}")
        raise
    
    pdf_file = io.BytesIO(response.content)
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    print("INFO: Successfully extracted text from PDF.")
    return text

def get_embeddings_from_api(texts: list[str]) -> np.ndarray:
    """Gets text embeddings from the Hugging Face API."""
    if not HF_TOKEN:
        raise ValueError("Hugging Face API token (HF_TOKEN) not set.")
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    if response.status_code != 200:
        raise Exception(f"Hugging Face API request failed with status {response.status_code}: {response.text}")
    
    embeddings = np.array(response.json())
    # The API returns a 3D array, we need 2D, so we average the inner dimension
    if len(embeddings.shape) == 3:
        embeddings = embeddings.mean(axis=1)
        
    return embeddings

def generate_answers_from_document(pdf_url: str, questions: list[str]) -> list[str]:
    """Main function to orchestrate the RAG process using APIs."""
    # 1. Get Text
    document_text = get_text_from_pdf(pdf_url)
    text_chunks = [para.strip() for para in document_text.split('\n\n') if len(para.strip()) > 100]
    if not text_chunks:
        text_chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
    print(f"INFO: Split document into {len(text_chunks)} chunks.")

    # 2. Get Embeddings for chunks via API
    print("INFO: Getting embeddings for text chunks via API...")
    chunk_embeddings = get_embeddings_from_api(text_chunks)
    print("INFO: Chunk embeddings received.")

    final_answers = []
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')

    for question in questions:
        print(f"INFO: Processing question: '{question}'")
        
        # 3. Get Embedding for the question via API
        question_embedding = get_embeddings_from_api([question])
        
        # 4. Find relevant chunks using cosine similarity
        similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
        top_k_indices = np.argsort(similarities)[-3:][::-1] # Get top 3 indices
        
        relevant_chunks = [text_chunks[i] for i in top_k_indices]
        context = "\n---\n".join(relevant_chunks)
        
        # 5. Generate Answer with LLM
        prompt = f"""
        You are an expert Q&A system. Your task is to answer the user's question based ONLY on the provided context.
        If the answer is not found in the context, state that clearly.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        response = llm.generate_content(prompt)
        final_answers.append(response.text.strip())
        print("INFO: Answer generated.")
        
    return final_answers