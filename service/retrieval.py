from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(level=logging.INFO)

logging.getLogger(
    "langchain.retrievers.multi_query"
).setLevel(logging.INFO)

logging.getLogger(
    "langchain_classic.retrievers.multi_query"
).setLevel(logging.INFO)


llm = ChatMistralAI(model="mistral-medium-latest")

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an enterprise retrieval specialist responsible for improving
document retrieval in a corporate RAG system.

The knowledge base contains:
- HR policies
- IT policies
- Security policies
- Employee handbooks
- Compliance documents
- Leave and benefits guidelines
- Device and access management procedures
- Internal operating procedures

Your task is to generate multiple alternative search queries that maximize
the chance of retrieving relevant policy documents.

Requirements:

1. Preserve the original intent.
2. Generate queries from different employee perspectives.
3. Include both formal policy terminology and casual employee language.
4. Include synonyms and alternate wording.
5. Expand abbreviations when useful.
6. Generate keyword-focused search variants.
7. Consider how HR, Legal, Compliance, Security, and IT teams might phrase the same topic.
8. If the question concerns a process, generate queries describing:
   - eligibility
   - requirements
   - procedure
   - exceptions
   - approvals

Generate exactly 7 search queries using the following strategies:

1. Employee conversational phrasing
2. Formal corporate policy phrasing
3. HR terminology
4. IT / technical terminology
5. Compliance / governance terminology
6. Keyword-heavy retrieval query
7. FAQ-style question

Return ONLY the queries.
One query per line.
No numbering.
No explanations.
"""
    ),
    (
        "human",
        """
Employee Question:
{question}
"""
    )
])
embeddings = MistralAIEmbeddings(model="mistral-embed")

def topKChunks(query):
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        prompt=QUERY_PROMPT
    )
    docs =  retriever.invoke(query)
    print(docs)
    return docs
if __name__ == "__main__":
    docs = topKChunks("what are the benifits")
    print(docs)