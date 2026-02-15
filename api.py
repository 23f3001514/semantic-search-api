"""
Semantic Search API with Re-ranking
Simple Flask API for the challenge
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import time

app = Flask(__name__)

# ============================================================================
# Global state
# ============================================================================
documents = []
embeddings = None
vocab = {}

# ============================================================================
# Embedding functions (mock - replace with OpenAI/Cohere)
# ============================================================================

def build_vocab(texts):
    """Build vocabulary from documents"""
    global vocab
    words = set()
    for text in texts:
        words.update(text.lower().split())
    vocab = {word: idx for idx, word in enumerate(sorted(words))}

def text_to_vector(text):
    """Convert text to vector"""
    words = text.lower().split()
    vector = np.zeros(len(vocab))
    
    for word in words:
        if word in vocab:
            vector[vocab[word]] += 1
    
    if np.linalg.norm(vector) > 0:
        vector = vector / np.linalg.norm(vector)
    
    # Project to 384 dimensions
    target_dim = 384
    if len(vocab) > target_dim:
        np.random.seed(42)
        projection = np.random.randn(len(vocab), target_dim) / np.sqrt(target_dim)
        vector = np.dot(vector, projection)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
    else:
        vector = np.pad(vector, (0, target_dim - len(vector)))
    
    return vector

# ============================================================================
# Search functions
# ============================================================================

def cosine_similarity_search(query_vec, k=8):
    """Find top-k most similar documents"""
    # Normalize
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Cosine similarity
    similarities = np.dot(doc_norms, query_norm)
    
    # Top-k
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

def rerank(query, doc_contents, scores, k=5):
    """Re-rank documents by relevance"""
    results = []
    
    for doc_content, orig_score in zip(doc_contents, scores):
        # Calculate relevance
        query_terms = set(query.lower().split())
        doc_terms = doc_content.lower().split()
        
        if not query_terms or not doc_terms:
            relevance = 0.0
        else:
            # Term frequency
            matches = sum(1 for term in doc_terms if term in query_terms)
            tf = matches / len(doc_terms)
            
            # Position score
            pos_scores = []
            for term in query_terms:
                try:
                    pos = doc_terms.index(term)
                    pos_scores.append(1.0 / (1 + pos / len(doc_terms)))
                except ValueError:
                    pos_scores.append(0.0)
            pos_score = np.mean(pos_scores) if pos_scores else 0.0
            
            # Phrase matching
            q_bigrams = set(zip(query.lower().split()[:-1], query.lower().split()[1:]))
            d_bigrams = set(zip(doc_terms[:-1], doc_terms[1:]))
            phrase = len(q_bigrams & d_bigrams) / max(len(q_bigrams), 1) if q_bigrams else 0.0
            
            relevance = 0.4 * tf + 0.3 * pos_score + 0.3 * phrase
        
        # Combine: 40% vector + 60% relevance
        final_score = 0.4 * orig_score + 0.6 * min(1.0, relevance)
        results.append((doc_content, final_score))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'Semantic Search API',
        'endpoints': {
            '/search': 'POST - Search with re-ranking',
            '/load': 'POST - Load documents (optional)',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'documents': len(documents),
        'ready': len(documents) > 0
    })

@app.route('/load', methods=['POST'])
def load_documents():
    """Load documents into the system"""
    global documents, embeddings
    
    try:
        data = request.get_json()
        docs = data.get('documents', [])
        
        if not docs:
            return jsonify({'error': 'No documents provided'}), 400
        
        documents = docs
        
        # Build embeddings
        texts = [d['content'] for d in documents]
        build_vocab(texts)
        embeddings = np.array([text_to_vector(t) for t in texts])
        
        return jsonify({
            'status': 'success',
            'loaded': len(documents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Main search endpoint"""
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        query = data.get('query', '')
        k = data.get('k', 8)
        do_rerank = data.get('rerank', True)
        rerank_k = data.get('rerankK', 5)
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        if len(documents) == 0:
            return jsonify({'error': 'No documents loaded'}), 400
        
        # Stage 1: Vector search
        query_vec = text_to_vector(query)
        initial = cosine_similarity_search(query_vec, k=k)
        
        if not initial:
            return jsonify({
                'results': [],
                'reranked': False,
                'metrics': {'latency': 0, 'totalDocs': len(documents)}
            })
        
        # Get documents
        doc_ids = [idx for idx, _ in initial]
        scores = [score for _, score in initial]
        doc_contents = [documents[idx]['content'] for idx in doc_ids]
        
        # Stage 2: Re-rank
        if do_rerank:
            reranked = rerank(query, doc_contents, scores, k=rerank_k)
            
            # Build results
            results = []
            for content, score in reranked:
                # Find the doc id
                doc_id = next(i for i, d in enumerate(documents) if d['content'] == content)
                results.append({
                    'id': doc_id,
                    'score': round(score, 4),
                    'content': content,
                    'metadata': documents[doc_id].get('metadata', {})
                })
        else:
            results = []
            for idx, score in initial[:rerank_k]:
                results.append({
                    'id': idx,
                    'score': round(score, 4),
                    'content': documents[idx]['content'],
                    'metadata': documents[idx].get('metadata', {})
                })
        
        elapsed = (time.time() - start_time) * 1000
        
        return jsonify({
            'results': results,
            'reranked': do_rerank,
            'metrics': {
                'latency': round(elapsed, 2),
                'totalDocs': len(documents)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# Startup
# ============================================================================

if __name__ == '__main__':
    # Auto-load documents if available
    try:
        with open('scientific_abstracts.json', 'r') as f:
            documents = json.load(f)
            texts = [d['content'] for d in documents]
            build_vocab(texts)
            embeddings = np.array([text_to_vector(t) for t in texts])
            print(f"✓ Auto-loaded {len(documents)} documents")
    except FileNotFoundError:
        print("⚠ No documents loaded. Use POST /load to add documents.")
    
    print("\n🔍 Semantic Search API Running")
    print("   URL: http://localhost:5000")
    print("   Endpoint: POST /search")
    print("\nExample:")
    print('  curl -X POST http://localhost:5000/search \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "machine learning", "k": 8, "rerank": true, "rerankK": 5}\'')
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)