from flask import Blueprint, render_template, jsonify

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/sample_queries')
def sample_queries():
    return jsonify([
        "What is quantum computing and how does it differ from classical computing?",
        "What are qubits, superposition, and entanglement in quantum computing?",
        "How could quantum computing impact cryptography and data security?",
        "What are some real-world applications of quantum computing?",
        "What are the main challenges and future trends in quantum computing?"
    ]) 