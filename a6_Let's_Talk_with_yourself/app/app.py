from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import torch
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

app = Flask(__name__)
CORS(app) 

# Serve the frontend
@app.route('/')
def home():
    return render_template("index.html")

# 🔹 Load embedding model
model_name = 'hkunlp/instructor-base'
embedding_model = HuggingFaceInstructEmbeddings(model_name=model_name)

# 🔹 Load FAISS vector store
vector_path = "./vector-store"
db_file_name = "nlp_stanford"

vectordb = FAISS.load_local(
    folder_path=os.path.join(vector_path, db_file_name),
    embeddings=embedding_model,
    index_name="nlp",
    allow_dangerous_deserialization=True
)
retriever = vectordb.as_retriever()

# 🔹 Load LLM Model
model_id = "lmsys/fastchat-t5-3b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="cpu")

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    model_kwargs={"temperature": 0, "repetition_penalty": 1.5}
)

# 🔹 Define PeteBot Prompt
prompt_template = """
Hey there! I’m KIDBOT, your AI-powered assistant, speaking as Tada Suttaket! 🎓🤖 
I am a Master's student at AIT, Thailand, and I can answer questions about my education, work experience, beliefs, and academic journey.

Ask me anything about:
My age, education, and major.
My work experience and industry involvement.
My thoughts on technology and culture.
My academic challenges and research interests.

If I don’t have an answer, I’ll be honest and say, 'Oops, I don’t know that one!'

{context}
Question: {question}
Answer:
""".strip()

PROMPT = PromptTemplate.from_template(prompt_template)

# 🔹 Question Generator Chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

# ✅ Wrap the pipeline for LangChain compatibility
llm = HuggingFacePipeline(pipeline=pipe)

question_generator = LLMChain(
    llm=llm,
    prompt=CONDENSE_QUESTION_PROMPT,
    verbose=True
)

# 🔹 Document Processing Chain
doc_chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    prompt=PROMPT,  # ✅ Uses the KIDBOT prompt
    verbose=True
)

# 🔹 Memory for Chat History
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 🔹 LangChain ConversationalRetrievalChain
chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,  
    combine_docs_chain=doc_chain,  
    memory=memory,
    return_source_documents=True,
    verbose=True,
    get_chat_history=lambda h : h
)

@app.route('/chat', methods=['POST'])
@app.route('/ask', methods=['POST'])  # Support both '/chat' and '/ask'
def chat():
    """Handles chatbot questions from the web app"""
    data = request.get_json()  # Use get_json() to handle None cases
    if not data or "question" not in data:
        return jsonify({"error": "Invalid request, 'question' key missing"}), 400

    question = str(data.get("question", "")).strip()  # Ensure input is a clean string

    print(f"\n🚀 Received question: {question}")

    # Get the response from the chain
    result = chain.invoke({"question": question, "chat_history": []})

    print(f"\n📌 Result from LLM: {result}")

    # Handle missing "answer" key safely
    answer = result.get("answer", "Sorry, I don't know the answer.")

    return jsonify({"answer": answer})  # Ensure JSON response format is correct

# 🔹 Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
