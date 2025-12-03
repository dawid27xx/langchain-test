from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# chroma is a local vector store cus my laptop old I can't download the proper ones

loader = TextLoader("data.txt")
documents = loader.load()

# splits document into chunks to help with embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)