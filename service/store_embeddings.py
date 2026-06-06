from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

import hashlib

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


def generate_chunk_id(doc: Document) -> str:
    source = doc.metadata.get("source_filename", "")
    page = doc.metadata.get("page_number", "")

    text = f"{source}|{page}|{doc.page_content.strip()}"

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_documents(chunks: list[Document]):
    ids = [generate_chunk_id(chunk) for chunk in chunks]

    existing_ids = set()

    try:
        existing = vector_store.get(ids=ids)
        existing_ids.update(existing["ids"])
    except Exception:
        pass

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids
        )

    print(
        f"Added {len(new_chunks)} chunks, "
        f"skipped {len(chunks) - len(new_chunks)} duplicates."
    )


if __name__ == "__main__":
    chunks = [
        Document(
            page_content="This is chunk 1",
            metadata={"source_filename": "file1.pdf", "page_number": 1}
        ),
        Document(
            page_content="This is chunk 2",
            metadata={"source_filename": "file1.pdf", "page_number": 2}
        )
    ]

    upsert_documents(chunks)