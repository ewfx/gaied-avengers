from flask import Flask, request, jsonify
import subprocess
import pandas as pd
from train_model import train_model_from_file

app = Flask(__name__)

# Endpoint to run the input.py script
@app.route('/run-script', methods=['POST'])
def run_script():
    # Get input data from the request
    input_data = request.json.get('input', '')

    try:
        # Call the input.py script and pass the input as an argument
        result = subprocess.run(
            ['python', 'classify_email.py', input_data],
            capture_output=True,
            text=True,
            check=True
        )
        # Return the output of the script
        return jsonify({'output': result.stdout.strip()})
    except subprocess.CalledProcessError as e:
        # Handle errors during script execution
        return jsonify({'error': e.stderr.strip()}), 500

# Endpoint to train a model (reads an Excel file)
@app.route('/train-model', methods=['POST'])
def train_model():
    # Check if a file is provided in the request
    if 'file' in request.files:
        file = request.files['file']
    else:
        # Fallback: Use a default Excel file from the current folder
        default_file_path = './traindata.xlsx'
        try:
            file = open(default_file_path, 'rb')  # Open the file in binary mode
        except FileNotFoundError:
            return jsonify({'error': f'Default file not found at {default_file_path}'}), 400

    try:
        # Call the function from model_logic.py
        accuracy, predictions = train_model_from_file(file)
        print(accuracy)
        print(predictions)
        return jsonify({
                "message": "Model trained successfully!"
            }), 200

    except Exception as e:
        print(f"Error: {e}")
        return None, None


if __name__ == '__main__':
    app.run(debug=True)