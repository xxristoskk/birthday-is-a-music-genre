import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)# Code to load ML model@app.route('/')

def home():    
     return render_template(index.html”)@app.route('/ml-model', methods=['POST'])
def run_model():
     # Code to use the trained model to make real time predictions
     return render_template(“index.html”, result)if __name__ == '__main__':
     app.run()
