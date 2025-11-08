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

## Core Components

### RAGPipeline Class

**Main Methods:**

- `chunk_text()` - Splits documents into retrievable chunks
- `create_knowledge_base()` - Processes documents and builds indices
- `similarity_search()` - Retrieves relevant chunks (semantic/keyword/hybrid)
- `generate_response()` - Generates answers using retrieved context
- `add_documents()` - Incrementally add new documents
- `save_knowledge_base()` - Persist knowledge base and embeddings
- `load_knowledge_base()` - Load previously saved data

### Search Methods

- **Semantic**: Vector similarity using embeddings
- **Keyword**: BM25-based ranking
- **Hybrid**: Combines semantic, keyword, and FAISS results

### Utils Module

`get_documents(doc_path)` - Loads all .txt files from a directory and returns list of document dictionaries.

## Configuration

Set in `main.py`:

```python
# Chunking
chunking_method = "recursive"  # "fixed", "recursive", or "document"
chunk_size = 256
overlap = 20

# Models (defaults to SmolLM2-360M-Instruct)
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
generator_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
```

## Requirements

- Python 3.8+
- langchain-text-splitters
- langchain-community
- scikit-learn
- faiss-cpu (or faiss-gpu)
- rank-bm25
- transformers
- torch
- numpy

Install all: `pip install -r requirements.txt`

## How It Works

1. **Document Loading**: Load .txt files from a directory
2. **Chunking**: Split documents using selected strategy
3. **Embedding**: Convert chunks to vectors using embedding model
4. **Indexing**: Build FAISS and BM25 indices for fast retrieval
5. **Retrieval**: Find relevant chunks using chosen method
6. **Generation**: Generate answer with LLM using retrieved context

## Example Workflow

```python
# 1. Initialize
rag = RAGPipeline()

# 2. Create knowledge base
docs = get_documents("./documents")
rag.create_knowledge_base(docs)

# 3. Query
context = rag.similarity_search("What is X?", method="hybrid", top_k=3)

# 4. Generate
response = rag.generate_response("What is X?", context)
print(response)

# 5. Add more docs later
new_docs = [...]
rag.add_documents(new_docs)
```

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{rag_implementation_2025,
  title={RAG Implementation from Scratch},
  author={Ani-404},
  year={2025},
  url={https://github.com/Ani-404/RAG-implementation-from-scratch}
}
```
