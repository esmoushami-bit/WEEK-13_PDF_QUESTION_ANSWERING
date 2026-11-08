
import pandas as pd
import chromadb

from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
PDF_PATH = DATASET_DIR / "sample.pdf"
DB_PATH = PROJECT_DIR / "chroma_db"

DATASET_DIR.mkdir(exist_ok=True)

# 1. Extract PDF text
reader = PdfReader(str(PDF_PATH))
extracted_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        extracted_text += f"\n--- Page {page_number} ---\n{text}"

(DATASET_DIR / "extracted_text.txt").write_text(
    extracted_text, encoding="utf-8"
)

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(extracted_text)

pd.DataFrame({
    "Chunk_ID": [f"chunk_{i}" for i in range(len(chunks))],
    "Content": chunks
}).to_csv(DATASET_DIR / "chunks.csv", index=False)

# 3. Generate embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    chunks, show_progress_bar=True
)

# 4. Store embeddings in ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))

try:
    client.delete_collection("ai_pdf_collection")
except Exception:
    pass

collection = client.create_collection(
    name="ai_pdf_collection",
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

# 5. Prepare questions
queries = [
    "What is Artificial Intelligence?",
    "What are the goals of AI?",
    "Explain machine learning.",
    "What is deep learning?",
    "What is Natural Language Processing?",
    "Explain computer vision.",
    "What is Generative AI?",
    "What is a Transformer architecture?",
    "What are Large Language Models?",
    "Explain Retrieval-Augmented Generation.",
    "What are text embeddings?",
    "What is ChromaDB?",
    "Explain semantic search.",
    "What is keyword-based search?",
    "What is Ollama?"
]

pd.DataFrame({"Query": queries}).to_csv(
    DATASET_DIR / "queries.csv", index=False
)

# 6. Semantic search
semantic_results = []
top_k = min(5, len(chunks))

for query in queries:
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    for rank, (chunk_id, document, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0]
        ), start=1
    ):
        semantic_results.append({
            "Query": query,
            "Rank": rank,
            "Chunk_ID": chunk_id,
            "Document": document,
            "Similarity_Score": 1 - distance
        })

semantic_df = pd.DataFrame(semantic_results)
semantic_df.to_csv(
    DATASET_DIR / "semantic_search_results.csv", index=False
)

# 7. Keyword search using TF-IDF
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(chunks)

keyword_results = []

for query in queries:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(
        query_vector, document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    for rank, index in enumerate(top_indices, start=1):
        keyword_results.append({
            "Query": query,
            "Rank": rank,
            "Chunk_ID": f"chunk_{index}",
            "Document": chunks[index],
            "Similarity_Score": scores[index]
        })

keyword_df = pd.DataFrame(keyword_results)
keyword_df.to_csv(
    DATASET_DIR / "keyword_search_results.csv", index=False
)

# 8. Compare search methods
comparison_df = pd.DataFrame({
    "Search_Method": ["Semantic Search", "Keyword Search"],
    "Average_Similarity_Score": [
        semantic_df["Similarity_Score"].mean(),
        keyword_df["Similarity_Score"].mean()
    ]
})

comparison_df.to_csv(
    DATASET_DIR / "search_comparison.csv", index=False
)

# 9. Generate answers using Ollama
llm = OllamaLLM(model="llama3.2")
answers = []

for query in queries:
    results = collection.query(
        query_embeddings=embedding_model.encode([query]).tolist(),
        n_results=top_k
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
    Answer using only the following PDF context.
    If the answer is unavailable, say:
    The answer is not available in the PDF.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = llm.invoke(prompt)

    answers.append({
        "Question": query,
        "Answer": response
    })

answers_df = pd.DataFrame(answers)
answers_df.to_csv(DATASET_DIR / "answers.csv", index=False)

print("PDF Question Answering System completed successfully!")
print("Total PDF pages:", len(reader.pages))
print("Total chunks:", len(chunks))
print("Total questions:", len(queries))
print("All CSV files saved in the dataset folder.")
