import os
from langchain_community.document_loaders import PyPDFDirectoryLoader

def infer_category(text: str) -> str:
    """
    Analyzes page content keywords to infer a document category.
    Modify or expand these rules based on your specific document types.
    """
    text_lower = text.lower()
    if "invoice" in text_lower or "total due" in text_lower or "billing" in text_lower:
        return "Financial/Invoice"
    elif "agreement" in text_lower or "contract" in text_lower or "terms" in text_lower:
        return "Legal/Contract"
    elif "resume" in text_lower or "cv" in text_lower or "education" in text_lower:
        return "HR/Resume"
    elif "agenda" in text_lower or "minutes" in text_lower or "meeting" in text_lower:
        return "Administrative/Meeting"
    else:
        return "General Reference"


def retrieve_docs(path):
    loader = PyPDFDirectoryLoader(
        path=path,
        recursive=True
    )

    docs = loader.load()
    # 2. Load documents from the directory
    print(f"Total pages loaded: {len(docs)}")

    # 3. Post-process and inject custom metadata
    for doc in docs:
        source_path = doc.metadata.get("source", "unknown")
        
        filename = os.path.basename(source_path)
        
        raw_page = doc.metadata.get("page", 0)
        human_readable_page = raw_page + 1
        
        inferred_cat = infer_category(doc.page_content)
        
        doc.metadata.clear()
        doc.metadata["source_filename"] = filename
        doc.metadata["page_number"] = human_readable_page
        doc.metadata["inferred_category"] = inferred_cat
    return docs


if (__name__ == '__main__'):
    docs = retrieve_docs("data")
    if docs:
        print("\n--- First Page Snippet ---")
        print(docs[0].page_content[:200])
        print("\n--- Metadata ---")
        import pprint
        pprint.pprint(docs[0].metadata)
