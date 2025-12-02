import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_groq import ChatGroq  
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# Configuration
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("🤖 Interview Agent")


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found! Please add GROQ_API_KEY to your .env file.")
    st.stop()

with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    selected_role = st.selectbox(
        "Select Job Role", 
        [
            "Python Developer", 
            "Data Scientist", 
            "Machine Learning Engineer", 
            "DevOps Engineer", 
            "Java Developer",
            "Product Manager",
            "Other"
        ]
    )
    
    if selected_role == "Other":
        job_role = st.text_input("Enter Custom Job Role", "Software Engineer")
    else:
        job_role = selected_role
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF (This runs locally)..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success("Ready!")
        initial_greeting = "Hello! I have reviewed your resume. I see you are applying for the " + job_role + " position. Shall we begin?"
        st.session_state.chat_history.append({"role": "assistant", "content": initial_greeting})
        st.rerun()

if st.session_state.vectorstore:
    #  Use Groq (Free Llama 3 model)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.1-8b-instant"
    )
    
    # Custom Prompt
    template = f"""You are an interviewer for a {job_role} role.
    Use the Context to ask relevant questions.
    Context: {{context}}
    History: {{chat_history}}
    Candidate: {{question}}
    Interviewer:"""
    
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer'),
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    # Chat UI
    for msg in st.session_state.chat_history:
        
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        
        else:
            role = "user" if msg.type == 'human' else "assistant"
            content = msg.content
            
        with st.chat_message(role):
            st.write(content)

    if user_input := st.chat_input("Type your answer..."):
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        
        response = qa_chain.invoke({"question": user_input})
        
        
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        with st.chat_message("assistant"):
            st.write(response["answer"])
else:
     st.info("Please upload a PDF resume to start.")
