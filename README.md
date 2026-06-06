stramlit run app.py

# Problem Statement
Scenario: You are tasked with building a scalable RAG backend for an internal corporate policy chatbot. The system must ingest hundreds of complex policy PDFs, chunk them intelligently to fit context limits without losing meaning, and retrieve highly relevant answers using an advanced retrieval strategy.
Requirements:
Robust Document Ingestion (Loaders & Metadata):
Write a function ingest_documents(directory_path) that reads multiple PDFs from a folder.
Production constraint: Ensure that metadata (source filename, page number, and an inferred "document category") is attached to every loaded document object. Implement error handling for corrupted files.
Semantic Chunking (Text Splitters):
Implement a RecursiveCharacterTextSplitter.
Production constraint: Configure the chunk size and overlap specifically for Mistral or OpenAI models. Add a custom separator list to ensure the splitter respects Markdown headers (#, ##) before falling back to paragraphs and sentences.
Vector Store Initialization (Embeddings & ChromaDB):
Initialize a persistent ChromaDB instance locally.
Production constraint: Write an upsert function that avoids duplicating document chunks if the same PDF is ingested twice. Use a hashing mechanism on the chunk content to generate unique IDs for the vector store.
Advanced Retrieval Engine:
Implement a MultiQueryRetriever.
Production constraint: Write the underlying prompt for the LLM that generates the query variations. Ensure the variations specifically target different phrasing an employee might use when asking about HR or IT policies.
Integration & Execution:
Wrap the components into a RAGPipeline class with a query(user_input) method.
Production constraint: Include standard logging (using Python's logging module) to track the time taken for retrieval and the number of chunks retrieved before the LLM generates the final response.