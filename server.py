from flask import Flask, send_from_directory, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Flask will look in the "templates" folder

# Route to serve JSON files
@app.route('/json/<path:filename>')
def serve_json(filename):
    return send_from_directory('json', filename)  # Serve JSON files from "json" folder

if __name__ == '__main__':
    app.run(debug=True, port=5000)