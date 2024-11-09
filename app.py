from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv  
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    pass