import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pypdf
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify

os.environ["GROQ_API_KEY"] = "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"

app = Flask(__name__)

def load_faiss_index():
    embedding_model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
    return FAISS.load_local("index", embedding_model, allow_dangerous_deserialization=True)

def build_qa_system(faiss_index):
    retriever = faiss_index.as_retriever(search_type="similarity", k=3)
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-8b-8192")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

docs = load_data("AuraDataset.pdf")
faiss_index = embedding_data(docs)
qa_chain = build_qa_system(faiss_index)

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.json.get('query', '')
    if query.lower() == "exit":
        return jsonify({"message": "Exiting..."}), 200

    print("💭 Generating answer...")
    result = qa_chain.run(query)
    return jsonify({"answer": result}), 200

if __name__ == "__main__":
    app.run(debug=True)
