import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from service.document_loader import retrieve_docs
from service.text_splitter import chunking
from service.store_embeddings import upsert_documents
from service.retrieval import topKChunks


CHROMA_PATH = "./chroma_langchain_db"

llm = ChatMistralAI(model="mistral-medium-latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant.

Use the provided context as the primary source of truth.

You may:
- Extract information from the context.
- Perform calculations using values found in the context.
- Infer reasonable conclusions from information in the context.
- Use general professional knowledge (such as tax rules, financial year definitions, salary calculations, employment terminology, and standard business practices) when required to interpret or calculate values based on information present in the context.

You must NOT:
- Invent facts that are not supported by the context.
- Assume values that are not present in the context.
- Answer questions that are completely unrelated to the context.

A question is answerable if the key facts required to answer it are present in the context, even if additional reasoning, calculations, or domain knowledge are required.

Examples:
- If the context contains a salary or CTC, you may estimate taxes using your knowledge of current tax rules.
- If the context contains dates, you may determine the applicable financial year.
- If the context contains compensation components, you may compute monthly, annual, pre-tax, or post-tax figures.
- If the context contains bonus information, you may include it in compensation calculations.

If the context does not contain the key facts required to answer the question, respond with exactly:

not in any pdf

Context:
{context}
"""),
    ("human", "{question}")
])

st.title("PDF RAG")

with st.sidebar:
    st.header("Database")

    if st.button("Delete All Documents"):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            st.success("All documents deleted.")
        else:
            st.info("No documents found.")

uploaded_files = st.file_uploader(
    "Select PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.button("Process PDFs"):
    with tempfile.TemporaryDirectory() as temp_dir:

        for file in uploaded_files:
            file_path = Path(temp_dir) / file.name

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        docs = retrieve_docs(temp_dir)
        texts = chunking(docs)
        upsert_documents(texts)

    st.success("Documents processed successfully.")

query = st.text_input("Ask any question from the docs provided")

if st.button("Submit") and query:

    retrieved_docs = topKChunks(query)

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    formatted_messages = prompt.format_messages(
        context=context_text,
        question=query
    )

    with st.spinner("Generating answer..."):
        ai_message = llm.invoke(formatted_messages)

    st.write(ai_message.content)