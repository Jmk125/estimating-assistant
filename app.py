from flask import Flask, request, jsonify
import os
import pandas as pd
import pdfplumber
import torch
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from fuzzywuzzy import fuzz

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper Functions to Connect to File Server and Extract Data
def get_files_from_server(server_path):
    files = []
    for root, dirs, file_names in os.walk(server_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            files.append(file_path)
    return files

# Extract data from Excel files
def extract_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

# Extract text from PDFs
def extract_pdf_data(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Prepare data for training the model
def prepare_data_for_training(files):
    dataset = []
    for file in files:
        if file.endswith(".pdf"):
            content = extract_pdf_data(file)
            # Example question-answer pairs based on the content
            qa_pairs = [
                {"context": content, "question": "What is the total cost of the project?", "answers": {"text": "50000", "answer_start": content.find("50000")}},
                {"context": content, "question": "How many units of concrete are required?", "answers": {"text": "200", "answer_start": content.find("200")}},
                # Add more question-answer pairs based on your file content
            ]
            dataset.extend(qa_pairs)
    return dataset

# Fine-tune the model on custom data
def fine_tune_model(train_data):
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    # Convert to Hugging Face's dataset format
    def preprocess_function(examples):
        questions = examples['question']
        contexts = examples['context']
        answers = examples['answers']['text']
        start_positions = examples['answers']['answer_start']

        encodings = tokenizer(questions, contexts, truncation=True, padding=True)
        encodings.update({'start_positions': start_positions, 'end_positions': start_positions + len(answers)})
        return encodings

    dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset
    )

    trainer.train()

    # Save the model to a path for later use
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

# Helper function to perform fuzzy search
def search_in_files(query, extracted_data):
    matches = []
    for record in extracted_data:
        for key, value in record.items():
            if fuzz.partial_ratio(query.lower(), str(value).lower()) > 70:
                matches.append(record)
                break
    return matches

# Load the fine-tuned model
def load_fine_tuned_model():
    model_name = "./fine_tuned_model"
    return pipeline("question-answering", model=model_name, tokenizer=model_name)

# Basic Flask Application with Query Route
@app.route("/")
def index():
    return "Hello, your chat service is running!"

# Route to extract and fine-tune model
@app.route("/train", methods=["POST"])
def train():
    # Specify the path to the file server
    server_path = r"Z:\CM PRECON PROJECTS\Schools"
    
    # Get all files from the server
    files = get_files_from_server(server_path)
    
    # Prepare the data for fine-tuning
    train_data = prepare_data_for_training(files)
    
    # Fine-tune the model
    fine_tune_model(train_data)
    
    return jsonify({"message": "Model fine-tuned successfully!"})

# Chat route to query the fine-tuned model
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get('question')
    
    # Load the fine-tuned model
    qa_pipeline = load_fine_tuned_model()

    # Specify the context (all extracted data)
    server_path = r"Z:\CM PRECON PROJECTS\Schools"
    files = get_files_from_server(server_path)

    extracted_data = []
    for file in files:
        if file.endswith(".xlsx"):
            extracted_data.extend(extract_excel_data(file))  # Append Excel data
        elif file.endswith(".pdf"):
            extracted_data.append({"content": extract_pdf_data(file)})  # Append PDF text as a single field

    # Combine the content into one large context
    context = " ".join([item['content'] for item in extracted_data if 'content' in item])

    # Use the model to answer the question
    result = qa_pipeline(question=question, context=context)
    
    return jsonify({"answer": result['answer']})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
