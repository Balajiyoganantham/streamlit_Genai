from flask import Blueprint, request, jsonify
from rag.rag_system import RAGSystem
from rag.PromptGenerator import PROMPTING_METHODS

bp = Blueprint('rag', __name__)
rag_system = RAGSystem()  # Initialize at import time

# Removed /initialize endpoint

@bp.route('/analyze_chunking', methods=['GET'])
def analyze_chunking():
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 400
    analysis = rag_system.get_chunking_analysis()
    return jsonify(analysis)

@bp.route('/prompting_methods', methods=['GET'])
def prompting_methods():
    return jsonify(PROMPTING_METHODS)

@bp.route('/query', methods=['POST'])
def query():
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 400
    data = request.get_json()
    question = data.get('question')
    method = data.get('method')
    prompt_method = data.get('prompt_method')
    custom_prompt = data.get('custom_prompt')
    if not question or not method:
        return jsonify({"error": "Missing question or method"}), 400
    result = rag_system.query_with_method(question, method, prompt_method, custom_prompt)
    return jsonify(result)

@bp.route('/compare_methods', methods=['POST'])
def compare_methods():
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 400
    data = request.get_json()
    question = data.get('question')
    prompt_method = data.get('prompt_method')
    custom_prompt = data.get('custom_prompt')
    if not question:
        return jsonify({"error": "Missing question"}), 400
    results = {}
    for method in rag_system.qa_chains.keys():
        results[method] = rag_system.query_with_method(question, method, prompt_method, custom_prompt)
    return jsonify(results) 