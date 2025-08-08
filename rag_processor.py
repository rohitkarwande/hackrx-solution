import os
import io
import pypdf
import requests
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# We will configure the API key in a later step. This line just prepares the code.
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

def get_text_from_pdf(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    print(f"INFO: Fetching PDF from URL...")
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()  # This will raise an error if the download fails
    except requests.RequestException as e:
        print(f"ERROR: Could not download PDF. {e}")
        raise
    
    # Read the PDF content from the response
    pdf_file = io.BytesIO(response.content)
    pdf_reader = pypdf.PdfReader(pdf_file)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
        
    print("INFO: Successfully extracted text from PDF.")
    return text


def generate_answers_from_document(pdf_url: str, questions: list[str]) -> list[str]:
    """
    This is the main function that orchestrates the entire RAG process.
    """
    # 1. Get the text from the PDF document
    document_text = get_text_from_pdf(pdf_url)
    if not document_text:
        return ["Could not process the document or the document is empty."] * len(questions)

    # 2. Chunk the text into smaller, manageable pieces
    text_chunks = [para.strip() for para in document_text.split('\n\n') if len(para.strip()) > 100]
    if not text_chunks:
         # Fallback to a simpler split if the double newline split fails
        text_chunks = [document_text[i:i+1000] for i in range(0, len(document_text), 1000)]
    
    print(f"INFO: Split document into {len(text_chunks)} chunks.")
    
    # 3. Create vector embeddings for each chunk using a Sentence Transformer model
    print("INFO: Loading embedding model (this may take a moment on first run)...")
    model_embed = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    print("INFO: Encoding text chunks into vectors...")
    chunk_embeddings = model_embed.encode(text_chunks, convert_to_tensor=False, show_progress_bar=True)
    
    # 4. Build a FAISS index for fast similarity search
    print("INFO: Building FAISS index for searching...")
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings, dtype=np.float32))
    print("INFO: FAISS index created successfully.")

    # 5. For each question, find relevant chunks and use the LLM to generate an answer
    final_answers = []
    print("INFO: Configuring the generative model...")
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')

    for question in questions:
        print(f"INFO: Processing question: '{question}'")
        
        # Find the top 3 most relevant text chunks from our document
        question_embedding = model_embed.encode([question])
        _, indices = index.search(np.array(question_embedding, dtype=np.float32), k=3)
        
        relevant_chunks = [text_chunks[i] for i in indices[0]]
        context = "\n---\n".join(relevant_chunks)
        
        # Create a specific prompt for the LLM
        prompt = f"""
        You are an expert Q&A system. Your task is to answer the user's question based ONLY on the provided context.
        Do not use any outside knowledge. If the answer cannot be found in the context, state that clearly.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        
        try:
            response = llm.generate_content(prompt)
            final_answers.append(response.text.strip())
            print("INFO: Answer generated.")
        except Exception as e:
            print(f"ERROR: Could not generate answer from LLM. {e}")
            final_answers.append("Error: Could not generate an answer.")
        
    return final_answers