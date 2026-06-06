from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from service.document_loader import retrieve_docs
from service.text_splitter import chunking
from service.store_embeddings import upsert_documents
from service.retrieval import topKChunks

llm = ChatMistralAI(model="mistral-medium-latest")


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context below. Do not use any outside knowledge.\n\n"
        "CRITICAL RULE: If the answer cannot be found in the provided context, "
        "or if the context is insufficient to answer, respond with exactly: "
        "'not in any pdf'. Do not explain, do not apologize, and do not add "
        "any other text.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])

path = input("provide folder path of pdf documents: ")
docs = retrieve_docs(path)
texts = chunking(docs)
upsert_documents(texts)
query = input("Ask any question from the docs provided: ")

retrieved_docs = topKChunks(query)



context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

formatted_messages = prompt.format_messages(
    context=context_text, 
    question=query
)

ai_message = llm.invoke(formatted_messages)
response = ai_message.content

print(response)
