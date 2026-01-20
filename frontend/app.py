import streamlit as st
import requests

st.title("OCR + Question Answering System")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Send image to OCR endpoint
    files = {"file": uploaded_file.getvalue()}
    ocr_response = requests.post("http://127.0.0.1:8000/ocr", files=files).json()
    
    text = ocr_response.get("extracted_text", "")
    st.text_area("Extracted Text", text, height=200)

    question = st.text_input("Ask a question about this text")
    
    if st.button("Get Answer") and question:
        qa_response = requests.post(
            "http://127.0.0.1:8000/qa",
            data={"question": question, "context": text}
        ).json()
        st.write("Answer:", qa_response.get("answer") or qa_response.get("error"))
