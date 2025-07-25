from sklearn.metrics import f1_score
import re
from rag.rag_system import RAGSystem

# Define evaluation questions and reference answers
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

# Helper: simple tokenization for F1
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

def main():
    rag_system = RAGSystem()
    f1_scores = []
    for item in EVAL_SET:
        try:
            result = rag_system.query_with_method(
                question=item['question'],
                method_name='fixed_size',  # or any default method
                prompt_method='default',
                custom_prompt=None
            )
            answer = result.get('answer', '')
            f1 = compute_f1(answer, item['reference'])
            f1_scores.append(f1)
            print(f"Q: {item['question']}\nPred: {answer}\nRef: {item['reference']}\nF1: {f1:.3f}\n{'-'*60}")
        except Exception as e:
            print(f"Error evaluating question: {item['question']}\n{e}\n{'-'*60}")
            f1_scores.append(0.0)
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"\nAverage F1 score: {avg_f1:.3f}")

if __name__ == "__main__":
    main() 