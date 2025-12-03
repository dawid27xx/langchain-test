import json

DB_FILE = "documents.json"

def save_documents(documents):
    data = [
        {
            "id": doc.metadata["id"],
            "title": doc.metadata["title"],
            "source": doc.metadata["source"],
            "date": doc.metadata["date"],
            "content": doc.page_content
        }
        for doc in documents
    ]
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_document_by_id(doc_id):
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    for doc in data:
        if doc["id"] == doc_id:
            return doc
    return None