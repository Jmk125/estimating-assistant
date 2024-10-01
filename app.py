from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple route to check if the server is running
@app.route("/")
def index():
    return "Hello, your chat service is running!"

# Placeholder route for querying the files (we will expand this later)
@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get('query')

    # We'll add the logic to search through local files here later
    return jsonify({"message": f"Received query: {query_text}"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
