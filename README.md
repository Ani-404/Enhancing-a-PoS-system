# RAG Implementation from Scratch

A **Retrieval-Augmented Generation (RAG)** pipeline implementation for document-based question answering. This project demonstrates building a complete RAG system using custom retrievers, embeddings, and LLM-based generators.

## Overview

This implementation combines semantic search, keyword-based retrieval, and text generation to answer queries based on your documents. It supports multiple chunking strategies and hybrid retrieval methods for accurate context retrieval.

## Key Features

- **Text Chunking**: Fixed, recursive, or document-based splitting strategies
- **Hybrid Retrieval**: Semantic search (embeddings), keyword search (BM25), and FAISS indexing
- **Multiple Models**: Swap embedding and generator models from Hugging Face
- **Knowledge Base Storage**: Save and load as JSON with embeddings as NumPy arrays
- **LLM Generation**: Generates responses using retrieved context

## Quick Start

### Installation

```bash
git clone https://github.com/Ani-404/RAG-implementation-from-scratch.git
cd RAG-implementation-from-scratch

pip install -r requirements.txt
```

### Usage

**Basic command line:**
```bash
python main.py --query "Your question here?"
```

**Custom models:**
```bash
python main.py \
  --query "Your question?" \
  --embedder "sentence-transformers/all-MiniLM-l6-v2" \
  --generator "HuggingFaceTB/SmolLM2-360M-Instruct" \
  --doc_path "./documents"
```

**Programmatic usage:**
```python
from rag_pipeline import RAGPipeline
from utils import get_documents

# Load documents
documents = get_documents(doc_path='./documents')

# Initialize pipeline
rag = RAGPipeline(
    embedding_model='sentence-transformers/all-MiniLM-l6-v2',
    generator_model='HuggingFaceTB/SmolLM2-360M-Instruct'
)

# Create knowledge base
rag.create_knowledge_base(documents, chunking_method='recursive', chunk_size=256, overlap=20)

# Search and generate
context = rag.similarity_search("Your question?", method='hybrid', top_k=3)
response = rag.generate_response("Your question?", context)
print(response)
```

## Project Structure

```
RAG-implementation-from-scratch/
├── main.py              # Entry point with CLI
├── rag_pipeline.py      # Core RAGPipeline class
├── utils.py             # Document loading utility
├── documents/           # Your .txt documents
└── requirements.txt     # Dependencies


```

