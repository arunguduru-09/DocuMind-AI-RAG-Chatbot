# DocuMind AI – RAG Chatbot

## Project Overview

DocuMind AI is an AI-powered PDF question-answering application built using Python, Streamlit, Sentence Transformers, Scikit-learn, and Ollama.

The application allows users to upload PDF documents, ask questions in natural language, and receive relevant answers based on document content.

## Key Features

- Upload and process PDF documents
- Extract text from PDFs using PyMuPDF
- Split documents into smaller text chunks
- Generate semantic embeddings using Sentence Transformers
- Retrieve relevant information using cosine similarity
- Generate answers using Ollama and Llama 3.2
- Display document names and page references
- Interactive user interface using Streamlit

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| Scikit-learn | Cosine similarity |
| Ollama | Local LLM execution |
| Llama 3.2:3b | Answer generation |

## RAG Architecture

```text
PDF Upload
    |
    v
Text Extraction
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
Similarity Search
    |
    v
Relevant Document Chunks
    |
    v
Ollama LLM
    |
    v
Generated Answer + Source References
```

## Installation

Clone the repository:

```bash
git clone https://github.com/arunguduru-09/DocuMind-AI-RAG-Chatbot.git
cd DocuMind-AI-RAG-Chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install streamlit pymupdf sentence-transformers scikit-learn ollama
```

Download the Ollama model:

```bash
ollama pull llama3.2:3b
```

Run the application:

```bash
streamlit run web_app.py
```

Note: Confirm the Streamlit entry-point filename and dependencies against the project source code before running.

## Example Use Cases

- Banking Policy and SOP Assistant
- HR Document Question Answering
- Educational Document Assistant
- Company Knowledge Base Assistant

## Future Improvements

- FAISS or ChromaDB integration
- FastAPI REST endpoints
- Multi-document comparison
- Chat history
- RAG evaluation
- Cloud deployment

## Author

**Arun Kumar**

GenAI Engineer | Python | RAG | LLM Applications

## Disclaimer

AI-generated answers should be verified against their original document sources.
