import streamlit as st
from rag.rag_system import RAGSystem
from rag.PromptGenerator import PROMPTING_METHODS
import time
import re

# Page configuration
st.set_page_config(
    page_title="RAG System Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stCode, .stMarkdown, .stJson {
        color: #fff !important;
        background: #222 !important;
    }
    .stTextArea textarea {
        color: #fff !important;
        background: #222 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sample-question {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .sample-question:hover {
        background-color: #e8eaf6;
        border-color: #667eea;
        cursor: pointer;
    }
    
    .method-card {
        background-color: #ffffff;
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-container {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .source-doc {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG system (cache to avoid reloading on every rerun)
@st.cache_resource
def get_rag_system():
    return RAGSystem()

# Initialize session state
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

rag_system = get_rag_system()

# Helper: Prompting methods
prompt_method_keys = list(PROMPTING_METHODS.keys())
prompt_method_labels = list(PROMPTING_METHODS.values())

# Chunking methods (from RAGSystem)
CHUNKING_METHODS = [
    ("fixed_size", "📝 Fixed Size", "Splits text into chunks of fixed character length"),
    ("sentence_splitter", "🔤 Sentence Splitter", "Splits text at sentence boundaries for natural breaks"),
    ("recursive", "🔄 Recursive", "Hierarchical splitting with multiple separators"),
]

# Sample/suggested questions
SAMPLE_QUERIES = [
    "What is quantum computing and how does it differ from classical computing?",
    "What are qubits, superposition, and entanglement in quantum computing?",
    "How could quantum computing impact cryptography and data security?",
    "What are some real-world applications of quantum computing?",
    "What are the main challenges and future trends in quantum computing?"
]

# Evaluation set (from evaluate.py)
EVAL_SET = [
    {
        'question': "What is quantum computing and how does it differ from classical computing?",
        'reference': "Quantum computing uses qubits that can exist in multiple states simultaneously, leveraging superposition and entanglement, unlike classical computers that use bits (0 or 1). This allows quantum computers to solve certain problems much faster than classical computers."
    },
    {
        'question': "What are qubits, superposition, and entanglement in quantum computing?",
        'reference': "Qubits are quantum bits that can represent both 0 and 1 at the same time (superposition). Entanglement is a property where qubits become linked and the state of one affects the other, enabling powerful quantum computations."
    },
    {
        'question': "How could quantum computing impact cryptography and data security?",
        'reference': "Quantum computers can break current encryption methods like RSA by factoring large numbers efficiently, which threatens data security. This drives research into quantum-resistant cryptography."
    },
    {
        'question': "What are some real-world applications of quantum computing?",
        'reference': "Quantum computing can be used in cryptography, optimization, drug discovery, materials science, and complex simulations that are difficult for classical computers."
    },
    {
        'question': "What are the main challenges and future trends in quantum computing?",
        'reference': "Challenges include scalability, error correction, and stability of qubits. Future trends involve overcoming these barriers, developing quantum-safe cryptography, and expanding applications in various fields."
    }
]

TOKENIZER = re.compile(r'\w+')
def tokenize(text):
    return TOKENIZER.findall(text.lower())

def compute_f1(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG System Explorer</h1>
    <p>Retrieval-Augmented Generation with Multiple Chunking Methods</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Method selection in sidebar
    
    st.markdown("### 📋 Chunking Method")
    
    # Create radio buttons with descriptions
    chunking_options = []
    for key, label, desc in CHUNKING_METHODS:
        chunking_options.append(f"{label}")
    
    selected_chunking = st.radio(
        "Choose chunking strategy:",
        chunking_options,
        help="Different methods for splitting documents into chunks"
    )
    
    # Get the actual method key
    chunking_method = None
    for key, label, desc in CHUNKING_METHODS:
        if f"{label}" == selected_chunking:
            chunking_method = key
            st.info(f"ℹ️ {desc}")
            break
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prompting method
    
    st.markdown("### 🎯 Prompting Technique")
    prompt_method_label = st.selectbox(
        "Select prompting strategy:",
        prompt_method_labels,
        help="Choose how to structure the prompt for the LLM"
    )
    prompt_method = prompt_method_keys[prompt_method_labels.index(prompt_method_label)]
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query history
    if st.session_state["query_history"]:
        st.markdown("### 📚 Recent Queries")
        with st.expander("View History", expanded=False):
            for i, (q, timestamp) in enumerate(reversed(st.session_state["query_history"][-5:])):
                if st.button(f"🔄 {q[:50]}...", key=f"history_{i}"):
                    st.session_state["question"] = q
                    st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Sample questions section
    st.markdown("## 💡 Sample Questions")
    st.markdown("Click on any question below to get started:")
    
    # Display sample questions in a more attractive way
    for i, question in enumerate(SAMPLE_QUERIES):
        if st.button(
            f"❓ {question}",
            key=f"sample_{i}",
            help="Click to use this sample question"
        ):
            st.session_state["question"] = question
            st.rerun()
    
    # Main query input
    st.markdown("## 🔍 Your Question")
    question = st.text_area(
        "Enter your question here:",
        value=st.session_state.get("question", ""),
        height=100,
        placeholder="Ask anything about the documents in your knowledge base...",
        key="question_input"
    )
    
    # Update session state
    st.session_state["question"] = question
    
    # Query button with enhanced styling
    col_query, col_analyze = st.columns([1, 1])
    
    with col_query:
        query_button = st.button(
            "🚀 Query RAG System",
            type="primary",
            use_container_width=True
        )
    
    with col_analyze:
        analyze_button = st.button(
            "📊 Analyze Methods",
            use_container_width=True
        )

with col2:
    # Method information card
    st.markdown("## 🛠️ Current Configuration")
    
    st.markdown(
        f"""
        <div style='background:#222; color:#fff; padding:1em; border-radius:8px; margin-bottom:1em;'>
            <b>📋 Chunking Method:</b> {selected_chunking}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Quick stats
    if hasattr(rag_system, 'get_stats'):
        try:
            stats = rag_system.get_stats()
            st.markdown("## 📈 System Stats")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{stats.get('total_docs', 'N/A')}</h3>
                    <p>Documents</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{stats.get('total_chunks', 'N/A')}</h3>
                    <p>Chunks</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            pass

# Handle query execution
if query_button:
    if question.strip():
        # Add to history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["query_history"].append((question, timestamp))
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Retrieving relevant documents...")
        progress_bar.progress(25)
        
        with st.spinner("Processing your query..."):
            time.sleep(0.5)  # Brief pause for UX
            status_text.text("🤖 Generating response...")
            progress_bar.progress(75)
            
            try:
                result = rag_system.query_with_method(
                    question=question,
                    method_name=chunking_method,
                    prompt_method=prompt_method,
                    custom_prompt=None
                )
                progress_bar.progress(100)
                status_text.text("✅ Query completed!")
                
                # Display results
                st.markdown("## 🎯 Results")
                
                # Answer section
                answer = result.get('answer', 'No answer')
                st.markdown(
                    f"""
                    <div style='background:#222; color:#fff; padding:1em; border-radius:8px; margin-bottom:1em;'>
                        <b>💬 Answer:</b><br>{answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Styled Source Documents section
                source_docs = result.get("source_documents", [])
                st.markdown(
                    f"""
                    <div style='background:#222; color:#fff; padding:1em; border-radius:8px; margin-bottom:1em;'>
                        <b>Source Documents</b><br>
                        Found {len(source_docs)} relevant document chunk{'s' if len(source_docs)!=1 else ''}:
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                for doc in source_docs:
                    st.code(doc["content"])
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    else:
        st.warning("⚠️ Please enter a question before querying.")

# Handle analysis
if analyze_button:
    st.markdown("## 📊 Chunking Methods Analysis")
    
    with st.spinner("Analyzing chunking methods..."):
        try:
            analysis = rag_system.get_chunking_analysis()
            
            # Display analysis in a more readable format
            if isinstance(analysis, dict):
                for method, data in analysis.items():
                    with st.expander(f"📋 {method.replace('_', ' ').title()}", expanded=True):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                st.metric(
                                    label=key.replace('_', ' ').title(),
                                    value=value
                                )
                        else:
                            st.write(data)
            else:
                st.json(analysis)
                
        except Exception as e:
            st.error(f"❌ Error analyzing methods: {str(e)}")

# --- Evaluation Section ---
st.markdown("---")
st.header("🔍 Evaluate RAG System on Standard Questions")
if st.button("Run Evaluation"):
    with st.spinner("Evaluating..."):
        f1_scores = []
        results = []
        for item in EVAL_SET:
            try:
                result = rag_system.query_with_method(
                    question=item['question'],
                    method_name='fixed_size',  # or any default method
                    prompt_method='zero_shot',
                    custom_prompt=None
                )
                answer = result.get('answer', '')
                f1 = compute_f1(answer, item['reference'])
                f1_scores.append(f1)
                results.append({
                    'question': item['question'],
                    'answer': answer,
                    'reference': item['reference'],
                    'f1': f1
                })
            except Exception as e:
                f1_scores.append(0.0)
                results.append({
                    'question': item['question'],
                    'answer': f'Error: {e}',
                    'reference': item['reference'],
                    'f1': 0.0
                })
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        st.markdown(
            f"""
            <div style='background:#222; color:#fff; padding:1em; border-radius:8px; margin-bottom:1em;'>
                <b>Average F1 score:</b> {avg_f1:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )
        for res in results:
            st.markdown(
                f"""
                <div style='background:#222; color:#fff; padding:1em; border-radius:8px; margin-bottom:1em;'>
                    <b>Q:</b> {res['question']}<br>
                    <b>Pred:</b> {res['answer']}<br>
                    <b>Ref:</b> {res['reference']}<br>
                    <b>F1:</b> {res['f1']:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 RAG System Explorer | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)