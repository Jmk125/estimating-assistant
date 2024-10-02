from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pdfplumber
from transformers import pipeline

app = Flask(__name__)

# Serve the chat HTML page
@app.route("/")
def chat_interface():
    return render_template("chat.html")

# Load the fine-tuned model
def load_fine_tuned_model():
    model_name = "./fine_tuned_model"  # Path where the fine-tuned model is saved
    return pipeline("question-answering", model=model_name, tokenizer=model_name)

# Route for handling user queries
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get('question')

    # Load the fine-tuned model
    qa_pipeline = load_fine_tuned_model()

    # Extract content from files and create a large context (PDFs, Excel, etc.)
    server_path = r"Z:\CM PRECON PROJECTS\Schools"  # Adjust path as needed
    files = get_files_from_server(server_path)

    extracted_data = []
    for file in files:
        if file.endswith(".xlsx"):
            extracted_data.extend(extract_excel_data(file))  # Extract Excel data
        elif file.endswith(".pdf"):
            extracted_data.append({"content": extract_pdf_data(file)})  # Extract PDF text as a single field

    # Combine the content into one large context
    context = " ".join([item['content'] for item in extracted_data if 'content' in item])

    # Use the model to answer the question
    result = qa_pipeline(question=question, context=context)
    
    return jsonify({"answer": result['answer']})

# Helper Functions to extract data from the files
def get_files_from_server(server_path):
    files = []
    for root, dirs, file_names in os.walk(server_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            files.append(file_path)
    return files

def extract_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

def extract_pdf_data(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
