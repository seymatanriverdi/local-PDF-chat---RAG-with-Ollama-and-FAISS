# PDF Chat RAG with Ollama, FAISS and Local LLM

This project implements a local Retrieval-Augmented Generation (RAG) pipeline for chatting with PDF documents.

## Features

- PDF text extraction
- Text chunking with overlap
- Embedding generation using SentenceTransformers
- FAISS vector search
- Semantic retrieval
- Local LLM generation with Ollama and Llama 3.2 1B
- Term-based context filtering for better answers
- Local RAG pipeline without cloud APIs

## Tech Stack

- Python
- FAISS
- SentenceTransformers
- Ollama
- Llama 3.2 1B
- pypdf
- NumPy

## RAG Pipeline

PDF  
→ Chunking  
→ Embedding  
→ FAISS Index  
→ Semantic Search  
→ Retrieved Context  
→ Ollama LLM  
→ Answer


## Project structure 

- `local pdf-chat-rag-ollama-faiss/`
    - `pdf_chat.py`
    - `requirements.txt`
    - `README.md`
    - `data/`
        - `put_your_pdf_here.txt`


```text
## Notes

This project runs locally using Ollama.  
The local LLM quality depends on the selected model.  
For better generation quality, a stronger local model or cloud LLM can be used.

## Example

Question:

```text
güverte nedir
Answer:
Güverte, bir platform veya yataklık olarak tanımlanabilir.

Install dependencies:
    pip install -r requirements.txt

Pull the Ollama model:
    ollama pull llama3.2:1b

Run the app:
    python pdf_chat.py

Place your PDF inside the **data/** folder.
Example:
data/
    your_document.pdf
Then update this line in `pdf_chat.py`:
```python
PDF_PATH = BASE_DIR / "data" / "your_document.pdf"
