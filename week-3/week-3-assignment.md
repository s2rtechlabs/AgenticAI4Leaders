# Week 3: LLMs Assignment

**Course**: Large Language Models and Transformers  
**Due Date**: [To be announced]  
**Total Points**: 100

---

## Assignment Overview

This assignment will test your understanding of Transformers, LLMs, attention mechanisms, embeddings, and tokenization through both theoretical questions and practical implementation tasks.

---

## Part 1: Theoretical Questions (40 points)

### Section A: Transformers and BERT (10 points)

**Q1.1** (3 points) Explain the key difference between BERT and GPT architectures. Why is BERT called "bidirectional" and how does this affect its use cases?

**Q1.2** (4 points) The attention mechanism uses the formula: `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`
- Explain what Q, K, and V represent
- Why do we divide by √d_k?
- What would happen if we didn't use the scaling factor?

**Q1.3** (3 points) List three advantages of Transformers over RNNs/LSTMs and explain why each is important for modern NLP tasks.

---

### Section B: Attention Mechanisms (12 points)

**Q2.1** (4 points) Compare and contrast the following attention mechanisms:
- Multi-Head Attention
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)

Include a table showing their trade-offs in terms of memory, speed, and quality.

**Q2.2** (4 points) Explain Paged Attention:
- What problem does it solve?
- How does it achieve near-zero memory waste?
- Give a concrete example with numbers showing memory savings

**Q2.3** (4 points) Flash Attention vs Paged Attention:
- What is the primary optimization of each?
- Can they be used together? If yes, what are the combined benefits?
- Which would you choose for: (a) Training, (b) Inference serving?

---

### Section C: Embeddings and Tokenization (10 points)

**Q3.1** (3 points) Explain the difference between:
- Static embeddings (Word2Vec)
- Contextual embeddings (BERT)

Provide an example showing how the word "bank" would be embedded differently in two sentences.

**Q3.2** (4 points) Tokenization Economics:
- Calculate the cost difference between GPT-4o and Groq for the following scenario:
  - 50,000 requests per month
  - Average 2000 input tokens per request
  - Average 500 output tokens per request
- Show your calculations and the percentage savings

**Q3.3** (3 points) Why is Byte Pair Encoding (BPE) preferred over word-level tokenization for modern LLMs? List at least three reasons with examples.

---

### Section D: LLM Lifecycle (8 points)

**Q4.1** (4 points) Describe the 7 stages of building an LLM. For each stage, mention:
- Primary objective
- Key personnel involved
- Approximate duration

**Q4.2** (4 points) In the LLM application lifecycle, explain the roles and responsibilities of:
- ML Engineer
- MLOps Engineer
- SRE Engineer

How do these roles collaborate during deployment?

---

## Part 2: Practical Implementation (40 points)

### Task 1: Tokenization Analysis (15 points)

**Objective**: Analyze and compare different tokenization strategies.

**Requirements**:

1. Install required libraries:
```python
pip install tiktoken transformers
```

2. Implement a tokenization comparison tool that:
   - Takes a text input
   - Tokenizes it using GPT-4 (tiktoken), BERT (WordPiece), and T5 (SentencePiece)
   - Displays:
     - Token count for each
     - Actual tokens
     - Character-to-token ratio
     - Estimated cost for GPT-4o and Groq

3. Test with at least 5 different text samples including:
   - English text
   - Text with numbers
   - Code snippet
   - Text with special characters
   - Non-English text (if possible)

4. Create a comparison table and write a 200-word analysis of your findings.

**Deliverables**:
- Python script (`tokenization_analysis.py`)
- Results table (CSV or markdown)
- Analysis document (PDF or markdown)

**Grading**:
- Code correctness: 7 points
- Comprehensive testing: 4 points
- Analysis quality: 4 points

---

### Task 2: BERT Embeddings and Similarity (15 points)

**Objective**: Use BERT to generate embeddings and analyze semantic similarity.

**Requirements**:

1. Create a dataset of at least 20 sentences covering:
   - 5 sentences about technology
   - 5 sentences about sports
   - 5 sentences about food
   - 5 sentences about weather

2. Generate BERT embeddings for all sentences

3. Compute cosine similarity matrix

4. Visualize embeddings using:
   - PCA (2D projection)
   - t-SNE (2D projection)

5. Create a heatmap of the similarity matrix

6. Answer these questions in your report:
   - Do similar topics cluster together?
   - Which visualization (PCA or t-SNE) better separates the topics?
   - Find the most similar and most dissimilar sentence pairs
   - What does this tell you about BERT's understanding of semantics?

**Deliverables**:
- Python notebook (`bert_embeddings.ipynb`)
- Visualizations (PNG files)
- Report (PDF, 300-400 words)

**Grading**:
- Code implementation: 7 points
- Visualizations: 4 points
- Analysis and insights: 4 points

---

### Task 3: Cost Optimization Analysis (10 points)

**Objective**: Analyze and optimize LLM API costs for a real-world scenario.

**Scenario**:
You're building a customer support chatbot that:
- Receives 100,000 queries per month
- Average query: 150 tokens
- Average response: 200 tokens
- Needs to support 3 response variations per query (for A/B testing)

**Requirements**:

1. Calculate monthly costs for:
   - GPT-4o
   - GPT-4o-mini
   - Groq (Llama-70B)
   - Groq (Mixtral)

2. Consider optimization strategies:
   - Caching common responses
   - Prompt compression
   - Using smaller models for simple queries
   - Batch processing

3. Create a cost optimization plan that:
   - Reduces costs by at least 50%
   - Maintains quality
   - Provides specific implementation steps

4. Create visualizations showing:
   - Cost comparison bar chart
   - Monthly cost projection with optimizations
   - Break-even analysis

**Deliverables**:
- Calculation spreadsheet (Excel or Google Sheets)
- Optimization plan (PDF, 400-500 words)
- Visualizations (PNG files)

**Grading**:
- Accurate calculations: 4 points
- Optimization strategy: 4 points
- Presentation and visualizations: 2 points

---

## Part 3: Research and Critical Thinking (20 points)

### Task 4: Attention Mechanism Deep Dive (10 points)

**Objective**: Research and explain a specific attention mechanism in detail.

**Choose ONE of the following**:
1. Rotary Position Embedding (RoPE)
2. ALiBi (Attention with Linear Biases)
3. Sliding Window Attention
4. Flash Attention 2

**Requirements**:

Write a comprehensive report (800-1000 words) covering:

1. **Background** (2 points)
   - What problem does it solve?
   - When and why was it developed?

2. **Technical Details** (4 points)
   - How does it work? (Include mathematical formulas if applicable)
   - Implementation details
   - Comparison with standard attention

3. **Performance Analysis** (2 points)
   - Benchmarks and metrics
   - Trade-offs (speed, memory, quality)

4. **Real-World Applications** (2 points)
   - Which models use it?
   - Use cases where it excels
   - Limitations

**Deliverables**:
- Research report (PDF)
- At least 5 credible references

**Grading**:
- Technical accuracy: 5 points
- Depth of analysis: 3 points
- Clarity and presentation: 2 points

---

### Task 5: LLM Application Design (10 points)

**Objective**: Design a complete LLM-powered application.

**Scenario**:
Design an LLM application for ONE of the following:
1. Code review assistant for GitHub
2. Medical diagnosis support system
3. Legal document analyzer
4. Educational tutoring platform
5. Content moderation system

**Requirements**:

Create a comprehensive design document (1000-1200 words) including:

1. **System Architecture** (3 points)
   - Component diagram
   - Data flow
   - Technology stack

2. **Model Selection** (2 points)
   - Which LLM(s) to use and why
   - Fine-tuning strategy
   - Fallback options

3. **Implementation Plan** (2 points)
   - Development phases
   - Team roles and responsibilities
   - Timeline estimate

4. **Cost and Performance** (2 points)
   - Cost estimation
   - Performance requirements
   - Optimization strategies

5. **Challenges and Solutions** (1 point)
   - Potential issues
   - Mitigation strategies

**Deliverables**:
- Design document (PDF)
- Architecture diagram (PNG or draw.io)
- Cost estimation spreadsheet

**Grading**:
- Completeness: 4 points
- Technical feasibility: 3 points
- Innovation and creativity: 2 points
- Presentation quality: 1 point

---

## Bonus Tasks (Optional, +10 points)

### Bonus 1: Implement Attention Visualization (5 points)

Create an interactive visualization tool that:
- Takes a sentence as input
- Shows attention weights for each layer and head
- Allows toggling between different layers
- Highlights which words attend to which

Use Plotly or similar for interactivity.

---

### Bonus 2: Build a Mini RAG System (5 points)

Implement a simple Retrieval-Augmented Generation system:
- Index a small document collection (10-20 documents)
- Use embeddings for retrieval
- Generate responses using an LLM API
- Compare responses with and without RAG

---

## Submission Guidelines

### Format
- All code should be well-commented and follow PEP 8 style guide
- Reports should be in PDF format
- Include a README.md with:
  - How to run your code
  - Dependencies and installation instructions
  - Brief description of each deliverable

### File Structure
```
week3_assignment_[your_name]/
├── README.md
├── part2/
│   ├── task1/
│   │   ├── tokenization_analysis.py
│   │   ├── results.csv
│   │   └── analysis.pdf
│   ├── task2/
│   │   ├── bert_embeddings.ipynb
│   │   ├── visualizations/
│   │   └── report.pdf
│   └── task3/
│       ├── cost_analysis.xlsx
│       ├── optimization_plan.pdf
│       └── visualizations/
├── part3/
│   ├── task4_research_report.pdf
│   └── task5_design_document.pdf
└── bonus/ (if applicable)
```

### Submission
- Zip the entire folder
- Name: `week3_assignment_[your_name].zip`
- Submit via [submission platform]

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part 1: Theory | 40 | Correctness, depth, clarity |
| Part 2: Practical | 40 | Code quality, completeness, analysis |
| Part 3: Research | 20 | Technical accuracy, insights, presentation |
| **Total** | **100** | |
| Bonus | +10 | Innovation, implementation quality |

---

## Academic Integrity

- You may discuss concepts with classmates, but all code and writing must be your own
- Properly cite all sources and references
- Use of AI assistants (ChatGPT, etc.) must be disclosed
- Plagiarism will result in zero points

---

## Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Groq API Documentation](https://console.groq.com/docs)

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)

### Tutorials
- Week 3 Class Notes (provided)
- Week 3 Jupyter Notebook (provided)

---

## Tips for Success

1. **Start Early**: Don't wait until the last minute
2. **Test Incrementally**: Test each component as you build
3. **Document Everything**: Good documentation helps you and the grader
4. **Ask Questions**: Use office hours or discussion forums
5. **Experiment**: Try different approaches and compare results
6. **Be Creative**: Especially in the design task, show your unique thinking

---

## FAQ

**Q: Can I use libraries not mentioned in the assignment?**  
A: Yes, but document why you chose them and ensure they're easy to install.

**Q: What if I can't access paid APIs like GPT-4?**  
A: Use the free tier or focus on open-source alternatives. Document your approach.

**Q: How detailed should the code comments be?**  
A: Enough that someone unfamiliar with your code can understand it.

**Q: Can I work in groups?**  
A: No, this is an individual assignment. However, you can discuss concepts.

**Q: What if my analysis contradicts the class notes?**  
A: That's fine! Explain your reasoning and provide evidence.

---

## Support

- **Office Hours**: [Times and location]
- **Discussion Forum**: [Link]
- **Email**: [Instructor email]
- **TA Support**: [TA contact]

---

**Good luck! This assignment will deepen your understanding of LLMs and prepare you for real-world applications.**

---

*Last Updated: [Date]*  
*Version: 1.0*
