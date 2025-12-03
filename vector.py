import uuid
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from document_db import save_documents
import os


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

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)