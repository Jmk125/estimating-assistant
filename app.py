from flask import Flask, request, jsonify
import os
import pandas as pd
import pdfplumber
from fuzzywuzzy import fuzz

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Step 3: Helper Functions to Connect to File Server and Extract Data

# Function to get all files from the file server
def get_files_from_server(server_path):
    files = []
    for root, dirs, file_names in os.walk(server_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            files.append(file_path)
    return files

# Function to extract data from Excel files
def extract_excel_data(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

# Function to extract text from PDFs
def extract_pdf_data(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 4: Helper function to perform fuzzy search
def search_in_files(query, extracted_data):
    matches = []
    for record in extracted_data:
        # Loop through each key-value pair in the record
        for key, value in record.items():
            # Perform fuzzy matching on the text in the files
            if fuzz.partial_ratio(query.lower(), str(value).lower()) > 70:
                matches.append(record)
                break
    return matches

# Step 2: Basic Flask Application with Query Route
@app.route("/")
def index():
    return "Hello, your chat service is running!"

# Route to process queries
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get('query')

    # Specify the path to the file server (replace with your actual path)
    server_path = "Z:\CM PRECON PROJECTS\Schools\"

    # Get all files from the server
    files = get_files_from_server(server_path)

    # Extract data from each file
    extracted_data = []
    for file in files:
        if file.endswith(".xlsx"):
            extracted_data.extend(extract_excel_data(file))  # Append Excel data
        elif file.endswith(".pdf"):
            extracted_data.append({"content": extract_pdf_data(file)})  # Append PDF text as a single field

    # Perform the search on the extracted data
    matches = search_in_files(query_text, extracted_data)
    
    if matches:
        return jsonify({"matches": matches})
    else:
        return jsonify({"message": "No matches found"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
