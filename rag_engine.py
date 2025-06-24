
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 🔹 Load local sentence transformer for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Load a local QA model (DistilBERT works well and is fast)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # or use distilbert or other
    tokenizer="google/flan-t5-base",
    max_length=512
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

def store_and_embed(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db.as_retriever()

def ask_question(question: str, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.run(question)
