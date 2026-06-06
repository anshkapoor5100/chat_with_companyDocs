from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_loader import retrieve_docs

def chunking(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",
            "\uff0c",
            "\u3001",
            "\uff0e",
            "\u3002",
            "",
            "#",
            "##"
        ],
        chunk_size=10000,
        chunk_overlap=1500,
    )
    texts = text_splitter.split_documents(docs)
    return texts

if (__name__ == '__main__'):
    docs = retrieve_docs("data")
    texts = chunking(docs)
    print(texts[0])
    print(texts[1])