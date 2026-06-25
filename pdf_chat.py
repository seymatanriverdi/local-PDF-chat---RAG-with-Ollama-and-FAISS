import ollama
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "data" / "denizcilikk.pdf"


def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search(query, chunks, embedding_model, index, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "distance": float(distance)
        })

    return results

def filter_context_for_term(query, context):
    term = query.lower().replace("nedir", "").replace("?", "").strip()

    lines = context.splitlines()

    for line in lines:
        if ":" in line and line.lower().startswith(term):
            return line

    return context

def generate_answer_with_ollama(context, query):
    prompt = f"""
    Sen bir PDF soru-cevap asistanısın.

    Kurallar:
    - Sadece bağlamdaki bilgiyi kullan.
    - Cevabı bağlamdaki ilgili tanımı neredeyse aynen alarak ver.
    - Bağlamda açık tanım varsa onu değiştirerek yorumlama.
    - Cevap bağlamda yoksa "Bu bilgi dokümanda bulunamadı." de.
    - Cevabı Türkçe, kısa ve net ver.

    Bağlam:
    {context}

    Soru:
    {query}

    Cevap:
    """

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def main():
    print("Reading PDF...")
    text = read_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating FAISS index...")
    index = build_faiss_index(chunks, embedding_model)

    print("PDF Chat RAG System Ready.")

    while True:
        query = input("\nAsk a question or type 'exit': ")

        if query.lower() == "exit":
            print("Program finished.")
            break

        results = search(query, chunks, embedding_model, index, top_k=3)

        print("\nRetrieved Contexts:\n")

        context_parts = []

        for i, item in enumerate(results, start=1):
            print(f"--- Result {i} | Distance: {item['distance']:.4f} ---")
            print(item["chunk"][:1000])
            print()

            context_parts.append(item["chunk"])

        context = "\n\n".join(context_parts)
        context = filter_context_for_term(query, context)
        answer = generate_answer_with_ollama(context, query)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()