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

# Prepare data for fine-tuning
def prepare_data_for_training(files):
    dataset = []
    for file in files:
        if file.endswith(".pdf"):
            content = extract_pdf_data(file)
            # Example question-answer pairs (you'll need to expand this for your data)
            qa_pairs = [
                {"context": content, "question": "What is the total cost of the project?", "answers": {"text": "50000", "answer_start": content.find("50000")}},
                {"context": content, "question": "How many units of concrete are required?", "answers": {"text": "200", "answer_start": content.find("200")}},
                # Add more question-answer pairs as you extract relevant data from files
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

    # Save the fine-tuned model to a directory
    model_dir = './fine_tuned_model'
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Log model save status
    if os.path.exists(model_dir):
        print(f"Model saved successfully at {model_dir}")
        print(f"Files in {model_dir}: {os.listdir(model_dir)}")  # List files in the directory
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
            server_path = r"C:\Users\Joe\Documents\assistanttest"  # Adjust the path for your data
            files = get_files_from_server(server_path)
            print(f"Files found: {files}")  # Log the files found
            train_data = prepare_data_for_training(files)
            fine_tune_model(train_data)
            return jsonify({"message": "Model fine-tuning complete!"})

        # Otherwise, treat it as a question for the model to answer
        print("Loading fine-tuned model...")  # Log model loading
        qa_pipeline = load_fine_tuned_model()

        # Extract content from files and create a large context (PDFs, Excel, etc.)
        server_path = r"C:\Users\Joe\Documents\assistanttest"  # Adjust path as needed
        files = get_files_from_server(server_path)
        print(f"Files found: {files}")  # Log the files found

        extracted_data = []
        for file in files:
            if file.endswith(".xlsx"):
                extracted_data.extend(extract_excel_data(file))  # Extract Excel data
            elif file.endswith(".pdf"):
                extracted_data.append({"content": extract_pdf_data(file)})  # Extract PDF text as a single field

        # Combine the content into one large context
        context = " ".join([item['content'] for item in extracted_data if 'content' in item])
        print(f"Context length: {len(context)} characters")  # Log context length

        # Use the model to answer the question
        result = qa_pipeline(question=question, context=context)
        print(f"Model result: {result}")  # Log the model result

        return jsonify({"answer": result['answer']})

    except Exception as e:
        print(f"Error occurred: {e}")  # Log any error that occurs
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
