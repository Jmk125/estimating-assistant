from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pdfplumber
import docx
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

app = Flask(__name__)

# Route to serve the chat HTML page
@app.route("/")
def chat_interface():
    return render_template("chat.html")

# Helper Functions for File Extraction
def get_files_from_server(server_path):
    files = []
    for root, dirs, file_names in os.walk(server_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            files.append(file_path)  # Add full file path
    return files

def extract_excel_data(file_path):
    try:
        # Specify engine based on file type
        if file_path.endswith(".xls"):
            df = pd.read_excel(file_path, engine="xlrd")
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file extension for {file_path}")
        
        if df.empty:
            raise ValueError(f"Excel file {file_path} is empty or invalid.")
        
        return df.to_dict(orient='records')

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  # Skip file if an error occurs

def extract_pdf_data(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        if not text:
            raise ValueError(f"PDF {file_path} is empty or invalid.")
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_docx_data(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        text = "\n".join(full_text)
        if not text:
            raise ValueError(f"DOCX {file_path} is empty or invalid.")
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Validate file structure and check extensions
def validate_files(file_list):
    valid_files = []
    for file_path in file_list:
        # Check if the file exists and is non-empty
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            continue
        
        if os.path.getsize(file_path) == 0:
            print(f"Error: File {file_path} is empty.")
            continue
        
        # Validate extensions
        if not file_path.endswith(('.txt', '.csv', '.xls', '.xlsx', '.pdf', '.doc', '.docx')):
            print(f"Error: Invalid file extension for {file_path}.")
            continue

        valid_files.append(file_path)
    
    return valid_files

# Prepare the data for training
def train_model_on_files(file_list):
    try:
        valid_files = validate_files(file_list)
        if not valid_files:
            print("No valid files found. Aborting training.")
            return "No valid files found. Training aborted."
        
        print(f"Starting training on {len(valid_files)} valid files...")
        
        train_data = []
        for file in valid_files:
            data = None
            if file.endswith(".xlsx") or file.endswith(".xls"):
                data = extract_excel_data(file)
            elif file.endswith(".txt"):
                with open(file, "r", encoding="utf-8") as f:
                    data = f.read()
            elif file.endswith(".pdf"):
                data = extract_pdf_data(file)
            elif file.endswith(".docx") or file.endswith(".doc"):
                data = extract_docx_data(file)
            
            if data is not None:
                # Prepare sample data for training
                qa_pairs = [{
                    "context": str(data),
                    "question": "What is the main content?",
                    "answers": {"text": "Example answer", "answer_start": str(data).find("Example answer")},
                }]
                train_data.extend(qa_pairs)

        if not train_data:
            print("No valid training data available after extraction.")
            return "No valid training data available."
        
        print(f"Training data prepared. Starting fine-tuning on {len(train_data)} samples...")
        fine_tune_model(train_data)
        print("Model training completed successfully.")
        return "Model fine-tuning complete!"
    
    except Exception as e:
        print(f"Error occurred during training: {e}")
        return str(e)

# Fine-tune the model on the prepared dataset
def fine_tune_model(train_data):
    model_name = "distilbert-base-uncased"  # Pre-trained model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    def preprocess_function(examples):
        try:
            questions = [example['question'] for example in examples]
            contexts = [example['context'] for example in examples]
            answers = [example['answers']['text'] for example in examples]
            start_positions = [example['answers']['answer_start'] for example in examples]

            encodings = tokenizer(questions, contexts, truncation=True, padding=True)
            encodings.update({
                'start_positions': start_positions,
                'end_positions': [start + len(answer) for start, answer in zip(start_positions, answers)]  # Calculate end positions
            })
            return encodings
        except Exception as e:
            raise ValueError(f"Error processing examples: {e}")

    dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    if dataset.num_rows == 0:
        raise ValueError("The dataset is empty. Aborting training.")

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset
    )

    trainer.train()

    model_dir = './fine_tuned_model'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    if os.path.exists(model_dir):
        print(f"Model saved successfully at {model_dir}")
    else:
        print(f"Failed to save model at {model_dir}")

# Load the fine-tuned model
def load_fine_tuned_model():
    model_dir = os.path.join(os.getcwd(), "fine_tuned_model").replace("\\", "/")

    if not os.path.exists(model_dir):
        raise OSError(f"Model directory {model_dir} not found. Ensure the model has been trained and saved.")

    print(f"Loading model from {model_dir}")
    
    return pipeline("question-answering", model=model_dir, tokenizer=model_dir)

# Route for handling user queries
@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json
        question = data.get('question')

        if question.lower() == "train":
            print("Triggering model training...")

            server_path = r"Z:\CM PRECON PROJECTS\Schools"
            files = get_files_from_server(server_path)

            if not files:
                return jsonify({"error": f"No files found in {server_path}."}), 400

            response_message = train_model_on_files(files)
            return jsonify({"message": response_message})

        print("Loading fine-tuned model...")
        qa_pipeline = load_fine_tuned_model()

        response = qa_pipeline({
            'context': "Provide some text context or extracted document data here",
            'question': question
        })

        return jsonify(response)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
