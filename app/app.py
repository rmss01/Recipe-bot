from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from mychatbot1 import ChatBot

# Define allowed extensions for audio files
ALLOWED_EXTENSIONS = set({'wav', 'mp3', 'm4a', 'mkv'})

def allowed_file(filename):
    """
    Check if the uploaded file is allowed based on its extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the Flask application
app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    """
    Render the index.html template for the home page.
    """
    return render_template('index.html')

@app.route("/audio/upload", methods=['POST'])
def upload_audio():
    """
    Handle audio file upload and process it through the chatbot service.
    """
    # Check if a file part is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'audio not provided'}), 400
    
    # Get the file from the request
    file = request.files['file']
    
    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    
    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Initialize the ChatBot and process the audio file
            chatbot = ChatBot()
            print("Original file path:", file_path)
            transcripted_audio = chatbot.process_audio(file_path)
            formatted_response = chatbot.format_response(transcripted_audio)
            print("Formatted Response:", formatted_response)
            
            # Generate the instructions audio file path
            instrucciones_audio_filename = filename + "_response.wav"
            instrucciones_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], instrucciones_audio_filename)
            print("Instructions audio path:", instrucciones_audio_path)
            chatbot.instrucciones_to_audio(formatted_response["instrucciones"], instrucciones_audio_path)
            
            # Return the formatted response and instructions audio URL
            return jsonify({
                'response': formatted_response,
                'instrucciones_audio_url': request.host_url + 'audio/download/' + instrucciones_audio_filename
            }), 200
        
        except Exception as e:
            # Return an error message if any exception occurs
            return jsonify({'error': str(e)}), 500

    # Return an error message if the file is not allowed
    return jsonify({'error': 'file not allowed'}), 400

@app.route("/audio/download/<filename>", methods=['GET'])
def download_file(filename):
    """
    Provide a route for downloading the audio file.
    """
    print(f"Attempting to download file: {os.path.abspath(app.config['UPLOAD_FOLDER']), filename}")
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename), 200

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
