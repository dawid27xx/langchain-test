"""
This file uses a local vector store and embeds a set of documents.
The documents will then be available to be looked up by the agent
"""

import uuid
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from document_db import save_documents
import os

# this file loads documents, creates some metadata data and encodes it in the vector store
def generate_unique_id():
    return str(uuid.uuid4())

docs_folder = "docs"
documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_folder, filename))
        doc = loader.load()[0]
        doc.metadata = {
            "id": generate_unique_id(),
            "title": filename.replace(".txt", "").capitalize(),
            "source": "Local File",
            "date": "2025-12-03"
        }
        documents.append(doc)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
save_documents(docs)

# here is where the embedding happens. We do not need to split up the query, etc.
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)