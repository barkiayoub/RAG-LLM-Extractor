# -*- coding: utf-8 -*-

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

local_path = "tst.pdf"

# Local PDF file uploads
if local_path:
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")

# Preview first page (optional)
data[0].page_content

#!pip install nomic

# Load model directly
from transformers import AutoModel
import os
import einops
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#token should be generated from Hugginface, then past your own
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Add to vector database
from langchain.vectorstores import Chroma
db = Chroma.from_documents(chunks, embeddings)

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="tiiuae/falcon-40b-instruct", trust_remote_code=True)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"max_length":10000, "max_new_tokens":100})

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
