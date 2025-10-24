# NLP & Deep Learning Concepts Glossary

## 1. Transformer Architecture
- **Transformers**: Neural network architecture that uses self-attention to process sequences in parallel. Replaces RNNs for NLP tasks.  
- **Encoder**: Part of a transformer that processes input sequences into contextual representations.  
- **Decoder**: Part of a transformer that generates output sequences, often using cross-attention on encoder outputs.  

## 2. Attention Mechanisms
- **Attention Mechanism**: Allows the model to focus on relevant parts of input when making predictions.  
- **Self-Attention**: Each token attends to all other tokens in the same sequence.  
- **Multi-Head Attention**: Uses multiple attention “heads” in parallel to capture different relationships.  
- **Grouped Query Attention**: Attention computed over grouped queries for efficiency or structure.  
- **Paged Attention**: Optimized attention mechanism for long sequences.  
- **Query, Key, Value Matrix (Q, K, V)**: Core components of attention:
  - **Query**: What we are looking for  
  - **Key**: Candidate tokens we compare against  
  - **Value**: Actual information we use for output  

## 3. Tokenization
- **Tokenization**: Splitting text into units (tokens) the model can understand.  
- **Word-by-word tokenization**: Each word is a token; simple but OOV prone.  
- **Subword tokenization**: Splits words into subwords (like `un + ##believable`).  
- **Character-level tokenization**: Each character is a token; handles OOV but long sequences.  
- **Byte Pair Encoding (BPE)**: Subword tokenization method based on merging frequent byte pairs.  
- **WordPiece**: Another subword method (used in BERT), splits words into frequent subwords.  

## 4. Text Preprocessing
- **Stemming**: Reduces words to their root form using rules (`running → run`).  
- **Lemmatization**: Reduces words to their dictionary form, considering part of speech (`better → good`).  
- **Parts of Speech Tagging (POS Tagging)**: Labels words as nouns, verbs, adjectives, etc.  

## 5. Embeddings & Representations
- **Embedding**: Dense vector representation of words or tokens capturing semantic meaning.  
- **Bag of Words (BoW)**: Represents text as a count vector of words; ignores order.  
- **Continuous Bag of Words (CBOW)**: Predicts a word from its context words; part of Word2Vec.  

## 6. Language Models
- **BERT**: Bidirectional Encoder Representations from Transformers; masked language model, uses self-attention.  
- **GPT**: Generative Pretrained Transformer; autoregressive, predicts next token sequentially.  
- **Autoregressive**: Generates outputs one token at a time, using previous tokens.  
- **Autoencoder**: Encodes input into latent representation and reconstructs it; can be used in NLP tasks.  

## 7. Neural Network Types
- **Recurrent Neural Networks (RNNs)**: Sequential models capturing temporal dependencies.  
- **Long Short Term Memory (LSTM)**: RNN variant solving vanishing gradient problem, remembers long-term dependencies.  
- **Gated Recurrent Units (GRU)**: Simplified LSTM with fewer parameters, faster but similar performance.  
