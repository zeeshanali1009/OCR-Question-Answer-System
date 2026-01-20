import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
import pytesseract
from PIL import Image
import io

# Load .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR endpoint
@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        extracted_text = pytesseract.image_to_string(image)
        return {"extracted_text": extracted_text}
    except Exception as e:
        return {"error": str(e)}

# Q/A endpoint
@app.post("/qa")
async def question_answer(
    question: str = Form(...),
    context: str = Form(...)
):
    try:
        context = context[:3000]  # avoid token overflow
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the context provided."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300
        )

        # Correctly fetch answer
        content = response.choices[0].message.content
        answer = content[0].text if isinstance(content, list) else content

        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
