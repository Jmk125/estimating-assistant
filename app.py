from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pdfplumber
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
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

def extract_pdf_data(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# File validation function
def validate_files(file_list):
    """
    Validate the list of files before training.
    - Check if files exist.
    - Check if files are non-empty.
    - Ensure expected extensions (e.g., .txt, .csv, .xlsx).
    """
    valid_files = []
    for file_path in file_list:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            continue
        
        # Check if the file is non-empty
        if os.path.getsize(file_path) == 0:
            print(f"Error: File {file_path} is empty.")
            continue
        
        # Check if the file extension is valid (add extensions you expect to process)
        if not file_path.endswith(('.txt', '.csv', '.xlsx')):
            print(f"Error: Invalid file extension for {file_path}.")
            continue

        # If all checks pass, add the file to the valid list
        valid_files.append(file_path)
    
    return valid_files

# Train the model on valid files
def train_model_on_files(file_list):
    """
    Train the model on valid files.
    """
    valid_files = validate_files(file_list)
    if not valid_files:
        print("No valid files found. Aborting training.")
        return "No valid files found. Aborting training."

    print(f"Starting training on {len(valid_files)} valid files...")
    
    # Prepare data for fine-tuning
    train_data = prepare_data_for_training(valid_files)
    
    # Fine-tune the model
    fine_tune_model(train_data)
    return "Training completed successfully!"

# Prepare data for fine-tuning
def prepare_data_for_training(files):
    dataset = []

    for file in files:
        if file.endswith(".txt"):
            # Extract text data from .txt files
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            # Example hardcoded question-answer pairs for demo purposes
            # In practice, you'd extract or generate these from your text data
            qa_pairs = [
                {
                    "context": content,
                    "question": "What is the main theme of the text?",
                    "answers": {"text": "Revenge", "answer_start": content.find("Revenge")},
                },
                {
                    "context": content,
                    "question": "Who is the main character?",
                    "answers": {"text": "Hamlet", "answer_start": content.find("Hamlet")},
                },
            ]
            dataset.extend(qa_pairs)

        elif file.endswith(".pdf"):
            content = extract_pdf_data(file)
            qa_pairs = [
                {"context": content, "question": "What is the total cost of the project?", "answers": {"text": "50000", "answer_start": content.find("50000")}},
                {"context": content, "question": "How many units of concrete are required?", "answers": {"text": "200", "answer_start": content.find("200")}},
            ]
            dataset.extend(qa_pairs)

    return dataset

# Fine-tune the model on the prepared dataset
def fine_tune_model(train_data):
    model_name = "distilbert-base-uncased"  # Pre-trained model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    # Convert to Hugging Face's dataset format
    def preprocess_function(examples):
        questions = examples['question']
        contexts = examples['context']
        answers = examples['answers']['text']
        start_positions = examples['answers']['answer_start']

        encodings = tokenizer(questions, contexts, truncation=True, padding=True)
        encodings.update({
            'start_positions': start_positions,
            'end_positions': [start + len(answer) for start, answer in zip(start_positions, answers)]  # Calculate end positions
        })
        return encodings

    dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Training Arguments with remove_unused_columns=False
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False  # Ensure no columns are removed automatically
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model to a directory
    model_dir = './fine_tuned_model'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Log model save status
    if os.path.exists(model_dir):
        print(f"Model saved successfully at {model_dir}")
        print(f"Files in {model_dir}: {os.listdir(model_dir)}")
    else:
        print(f"Failed to save model at {model_dir}")

# Load the fine-tuned model
def load_fine_tuned_model():
    model_dir = os.path.join(os.getcwd(), "fine_tuned_model").replace("\\", "/")

    # Check if the directory exists before loading the model
    if not os.path.exists(model_dir):
        raise OSError(f"Model directory {model_dir} not found. Ensure the model has been trained and saved.")

    print(f"Loading model from {model_dir}. Contents: {os.listdir(model_dir)}")
    
    # Load the fine-tuned model and tokenizer from the directory
    return pipeline("question-answering", model=model_dir, tokenizer=model_dir)

# Route for handling user queries
@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json
        question = data.get('question')

        # If the user types "train", trigger the training process
        if question.lower() == "train":
            print("Triggering model training...")  # Log training start

            # Update the path to the correct folder that contains subdirectories
            server_path = r"Z:\CM PRECON PROJECTS\Schools"  # Adjust path as necessary
            
            # Call the recursive file search
            files = get_files_from_server(server_path)
            
            # Check if files were found
            if not files:
                print(f"No files found in {server_path} or its subdirectories.")
                return jsonify({"error": f"No files found in {server_path} or its subdirectories."}), 400
            else:
                print(f"Files found: {files}")  # Log found files to the console

            # Validate and train the model on valid files
            response_message = train_model_on_files(files)
            return jsonify({"message": response_message})

        # Otherwise, treat it as a question for the model to answer
        print("Loading fine-tuned model...")  # Log model loading
        qa_pipeline = load_fine_tuned_model()

        # (rest of the code for answering a question)
    
    except Exception as e:
        print(f"Error occurred: {e}")  # Log any error that occurs
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
