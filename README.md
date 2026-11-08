# PDF Question Answering System

## 1. Project Overview
This project develops a PDF Question Answering System using LangChain, ChromaDB, Sentence Transformers, and Ollama.

It extracts text from PDF documents, converts the text into chunks, generates embeddings, retrieves relevant information, and generates answers using a Large Language Model.

## 2. Objectives
- Extract text from PDF documents.
- Split extracted text into smaller chunks.
- Generate text embeddings.
- Store embeddings in ChromaDB.
- Perform semantic search and keyword search.
- Retrieve the Top 5 relevant documents for each query.
- Generate answers using Ollama (Llama 3.2).

## 3. Technologies Used
- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama (Llama 3.2)
- PyPDF
- Pandas
- Scikit-learn
- Jupyter Notebook

## 4. Project Workflow
1. Load and extract text from a PDF.
2. Divide the extracted text into chunks.
3. Generate embeddings using Sentence Transformers.
4. Store the embeddings in ChromaDB.
5. Perform semantic search using vector similarity.
6. Perform keyword search using TF-IDF.
7. Compare the search results.
8. Generate answers using Ollama.

## 5. Results
- PDF pages processed: 20
- Text chunks created: 90
- Questions processed: 15
- Top 5 results retrieved for each question.
- Semantic and keyword search results saved as CSV files.
- Answers generated using Llama 3.2.

## 6. Output Files
- extracted_text.txt
- chunks.csv
- queries.csv
- semantic_search_results.csv
- keyword_search_results.csv
- search_comparison.csv
- answers.csv
- observation_report.txt

## 7. Conclusion
The project successfully demonstrates PDF text extraction, vector embeddings, semantic search, keyword search, and question answering using a Large Language Model.