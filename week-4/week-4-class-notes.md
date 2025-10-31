# Week 4: RAG (Retrieval-Augmented Generation) - Class Notes

## Session Overview
**Topic**: Enterprise Knowledge Systems with RAG  
**Duration**: 2-3 hours  
**Format**: Lecture + Hands-on Exercise

---

## Part 1: Introduction to RAG (30 minutes)

### What is RAG?

**Definition**: Retrieval-Augmented Generation combines information retrieval with text generation to produce accurate, grounded responses.

**The RAG Pipeline:**
```
User Query → Embedding → Vector Search → Retrieve Docs → Augment Prompt → LLM → Response
```

### Why RAG Matters

**Problems RAG Solves:**
1. **Hallucinations**: LLMs making up facts
2. **Outdated Knowledge**: Training data cutoff dates
3. **Domain Specificity**: Lack of specialized knowledge
4. **Privacy**: Need to keep data local

**Real-World Impact:**
- Customer support automation
- Internal knowledge management
- Technical documentation search
- Legal and compliance queries

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Cost** | Low (no retraining) | High (GPU hours) |
| **Updates** | Instant (add documents) | Slow (retrain model) |
| **Accuracy** | High (grounded in docs) | Variable |
| **Privacy** | Good (local data) | Risky (data in model) |
| **Use Case** | Dynamic knowledge | Behavior/style changes |

---

## Part 2: Embeddings Deep Dive (30 minutes)

### What Are Embeddings?

**Concept**: Dense vector representations that capture semantic meaning

**Example:**
```python
"cat" → [0.2, -0.5, 0.8, ..., 0.3]  # 384 dimensions
"dog" → [0.3, -0.4, 0.7, ..., 0.2]  # Similar vector!
```

### How Embeddings Work

1. **Text Input**: "The cat sits on the mat"
2. **Tokenization**: ["The", "cat", "sits", "on", "the", "mat"]
3. **Neural Network**: Transformer encoder
4. **Output**: 384-dimensional vector

### Similarity Metrics

**Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)
Range: -1 to 1 (higher = more similar)
```

**Distance Metrics:**
- **Cosine**: Best for semantic similarity
- **Euclidean**: Geometric distance
- **Dot Product**: Fast but not normalized

### Popular Embedding Models

1. **Sentence Transformers** (Open Source)
   - `all-MiniLM-L6-v2`: Fast, 384 dims
   - `all-mpnet-base-v2`: Better quality, 768 dims

2. **OpenAI Embeddings** (Commercial)
   - `text-embedding-ada-002`: 1536 dims
   - High quality, pay per token

3. **Cohere Embeddings** (Commercial)
   - `embed-english-v3.0`: Multilingual support

---

## Part 3: Vector Databases (30 minutes)

### Why Vector Databases?

**Traditional Databases:**
- Store structured data (rows, columns)
- Search by exact match or SQL queries
- Not designed for similarity search

**Vector Databases:**
- Store high-dimensional vectors
- Fast similarity search (ANN algorithms)
- Optimized for embeddings

### Vector Database Options

#### 1. ChromaDB (Used in Exercise)
```python
# In-memory, simple, great for learning
client = chromadb.Client()
collection = client.create_collection("docs")
```

**Pros:**
- Easy to use
- No setup required
- Good for prototyping

**Cons:**
- In-memory only (data lost on restart)
- Not for production scale

#### 2. FAISS (Facebook AI)
```python
# High-performance similarity search
import faiss
index = faiss.IndexFlatL2(dimension)
```

**Pros:**
- Very fast
- Handles billions of vectors
- Free and open source

**Cons:**
- Lower-level API
- Requires more setup

#### 3. Pinecone (Managed Service)
```python
# Cloud-hosted vector database
import pinecone
pinecone.init(api_key="...")
```

**Pros:**
- Fully managed
- Scales automatically
- Production-ready

**Cons:**
- Costs money
- Data in cloud

#### 4. Weaviate, Qdrant, Milvus
- Open source alternatives
- Self-hosted or cloud
- Production-grade features

### Indexing Algorithms

**HNSW (Hierarchical Navigable Small World):**
- Graph-based algorithm
- Fast approximate search
- Good recall/speed tradeoff

**IVF (Inverted File Index):**
- Clustering-based
- Faster for very large datasets
- Slightly lower recall

---

## Part 4: Building RAG Systems (45 minutes)

### RAG Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Embed Query     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Vector Search   │
│ (Top K docs)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Build Context   │
│ (Retrieved docs)│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ LLM Generation  │
│ (With context)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Final Answer    │
└─────────────────┘
```

### Key Components

#### 1. Document Chunking
```python
# Split long documents into chunks
chunk_size = 500  # tokens
overlap = 50      # token overlap
```

**Strategies:**
- **Fixed-size**: Simple, consistent
- **Semantic**: Split by paragraphs/sections
- **Recursive**: Hierarchical splitting

#### 2. Metadata Management
```python
metadata = {
    "source": "manual.pdf",
    "page": 42,
    "category": "technical",
    "date": "2024-01-15"
}
```

**Benefits:**
- Filter by source/category
- Track provenance
- Enable hybrid search

#### 3. Retrieval Parameters
```python
# How many documents to retrieve?
k = 3  # Top 3 most relevant

# Similarity threshold?
min_similarity = 0.7  # Only if > 70% similar
```

#### 4. Context Building
```python
# Combine retrieved docs into context
context = "\n\n".join([
    f"Document {i}: {doc}"
    for i, doc in enumerate(retrieved_docs)
])
```

### Prompt Engineering for RAG

**Basic Template:**
```
You are a helpful assistant. Answer based on the context.

Context:
{retrieved_documents}

Question: {user_question}

Answer:
```

**Advanced Template:**
```
You are an expert assistant. Use the provided context to answer.
If the context doesn't contain the answer, say "I don't have enough information."
Cite sources when possible.

Context:
{retrieved_documents}

Question: {user_question}

Instructions:
- Be concise and accurate
- Use information from context only
- Cite document numbers [1], [2], etc.

Answer:
```

---

## Part 5: Advanced RAG Patterns (30 minutes)

### 1. Hierarchical RAG

**Concept**: Multi-level retrieval
```
Query → Retrieve Summaries → Select Relevant Sections → Retrieve Details
```

**Use Case**: Large document collections

### 2. GraphRAG

**Concept**: Build knowledge graphs from documents
```
Documents → Extract Entities/Relations → Build Graph → Graph Traversal → Context
```

**Benefits:**
- Capture relationships
- Multi-hop reasoning
- Better context understanding

### 3. Agentic RAG

**Concept**: AI agents that autonomously retrieve
```
Query → Agent Plans → Multiple Retrievals → Synthesize → Answer
```

**Features:**
- Multi-step reasoning
- Tool use (search, calculate, etc.)
- Adaptive retrieval

### 4. Hybrid Search

**Concept**: Combine semantic + keyword search
```
Query → [Vector Search + BM25] → Merge Results → Rerank → Top K
```

**Formula:**
```
score = α × semantic_score + (1-α) × keyword_score
```

### 5. Re-ranking

**Concept**: Improve initial retrieval
```
Initial Retrieval (k=20) → Cross-Encoder Rerank → Top K (k=3)
```

**Models:**
- Cross-encoders (slower but accurate)
- Learned rerankers
- Diversity-based selection

---

## Part 6: Enterprise Considerations (20 minutes)

### Data Preparation

**Best Practices:**
1. **Clean Data**: Remove duplicates, fix formatting
2. **Normalize**: Consistent structure
3. **Enrich Metadata**: Add categories, dates, sources
4. **Version Control**: Track document versions

### Performance Optimization

**Chunking:**
- Optimal size: 200-500 tokens
- Use overlap: 10-20% of chunk size
- Consider document structure

**Embedding:**
- Choose model based on use case
- Batch processing for efficiency
- Cache embeddings

**Retrieval:**
- Tune k (number of results)
- Use metadata filtering
- Implement caching

### Security & Privacy

**Access Control:**
```python
# Filter by user permissions
results = collection.query(
    query_texts=[query],
    where={"access_level": user.level}
)
```

**Data Encryption:**
- Encrypt at rest
- Secure transmission
- Audit logging

**Deployment Options:**
- On-premise (maximum privacy)
- Private cloud (VPC)
- Managed service (convenience)

### Monitoring & Evaluation

**Metrics to Track:**
1. **Retrieval Quality**
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)

2. **Generation Quality**
   - Answer relevance
   - Factual accuracy
   - Citation accuracy

3. **System Performance**
   - Latency (p50, p95, p99)
   - Throughput (queries/sec)
   - Cost per query

---

## Part 7: Case Study - Siemens (15 minutes)

### Challenge
- 10M+ pages of technical documentation
- Multiple languages and formats
- Engineers spending hours searching
- Need for precise, accurate answers

### Solution Architecture

**Data Pipeline:**
```
PDFs/Docs → Extract Text → Clean → Chunk → Embed → Index
```

**Retrieval:**
```
Query → Translate (if needed) → Embed → Search → Rerank → Top 5
```

**Generation:**
```
Context + Query → GPT-4 → Answer + Citations
```

### Key Innovations

1. **Custom Chunking**: Respect document structure
2. **Fine-tuned Embeddings**: Domain-specific training
3. **Hierarchical Retrieval**: Summary → Detail
4. **Multi-lingual**: Support 12 languages
5. **On-premise**: Data privacy compliance

### Results

**Quantitative:**
- 90% reduction in search time
- 85% answer accuracy
- 10,000+ daily queries
- <2 second response time

**Qualitative:**
- Improved engineer productivity
- Reduced support tickets
- Better knowledge sharing
- Faster onboarding

---

## Key Takeaways

### What We Learned

1. ✅ **RAG combines retrieval + generation** for accurate responses
2. ✅ **Embeddings capture semantic meaning** in vectors
3. ✅ **Vector databases enable fast similarity search**
4. ✅ **Chunking and metadata are crucial** for quality
5. ✅ **Advanced patterns** (GraphRAG, Agentic) for complex use cases

### Best Practices

1. **Start Simple**: Basic RAG before advanced patterns
2. **Iterate**: Test and improve retrieval quality
3. **Monitor**: Track metrics and user feedback
4. **Optimize**: Balance quality, speed, and cost
5. **Secure**: Implement proper access controls

### Common Pitfalls

❌ **Chunks too large**: Poor retrieval precision  
❌ **No metadata**: Can't filter effectively  
❌ **Wrong embedding model**: Poor semantic matching  
❌ **No reranking**: Suboptimal context  
❌ **Ignoring privacy**: Security vulnerabilities  

---

## Hands-On Exercise

**Objective**: Build a working RAG system with ChromaDB

**Steps:**
1. Set up ChromaDB and embeddings
2. Create knowledge base with AI/ML documents
3. Implement semantic search
4. Build RAG query function
5. Test with real questions
6. Compare RAG vs non-RAG responses

**Expected Outcome:**
- Understand RAG pipeline end-to-end
- See embeddings in action
- Experience semantic search
- Build confidence for real projects

---

## Additional Resources

### Documentation
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval" (Karpukhin et al., 2020)

### Tools
- **LangChain**: RAG framework
- **LlamaIndex**: Data framework for LLMs
- **Haystack**: NLP framework with RAG

---

## Q&A Topics

**Common Questions:**

1. **When to use RAG vs fine-tuning?**
   - RAG: Dynamic knowledge, frequent updates
   - Fine-tuning: Behavior/style changes

2. **How many documents to retrieve?**
   - Start with k=3-5
   - Tune based on context window and quality

3. **Which embedding model to choose?**
   - Sentence Transformers: Free, good quality
   - OpenAI: Best quality, costs money
   - Domain-specific: Fine-tune if needed

4. **How to handle long documents?**
   - Chunk with overlap
   - Use hierarchical retrieval
   - Consider summarization

5. **How to evaluate RAG quality?**
   - Human evaluation
   - Automated metrics (precision, recall)
   - A/B testing with users

---

**Next Week**: Advanced AI Agents and Tool Use
