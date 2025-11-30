import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_groq import ChatGroq  # Replaces ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings # Replaces OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Interviewer (Free Ver)", layout="wide")
st.title("🤖 Interview Agent (Powered by Groq)")


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found! Please add GROQ_API_KEY to your .env file.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    job_role = st.text_input("Job Role", "Python Developer")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# --- HELPER FUNCTIONS ---
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

# --- MAIN LOGIC ---
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
        role = "user" if msg.type == 'human' else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if user_input := st.chat_input("Type your answer..."):
        with st.chat_message("user"):
            st.write(user_input)
        
        response = qa_chain.invoke({"question": user_input})
        
        with st.chat_message("assistant"):
            st.write(response["answer"])
else:
     st.info("Please upload a PDF resume to start.")


























# import os
# import streamlit as st
# from pypdf import PdfReader
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_classic.chains import ConversationalRetrievalChain
# from langchain_classic.memory import ConversationBufferMemory
# from langchain_classic.prompts import PromptTemplate

# load_dotenv()
# api_key=os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("API Key not found! Please create a .env file with OPENAI_API_KEY.")
#     st.stop()

# # --- CONFIGURATION ---
# # Replace with your actual key
# # --- UI SETUP ---
# st.set_page_config(page_title="AI Interviewer", layout="wide")
# st.title("🤖 Intelligent Interview Agent")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("Interview Setup")
#     job_role = st.selectbox(
#         "Select Role:",
#         ["Python Developer", "Data Scientist", "DevOps Engineer", "Marketing Manager"]
#     )
#     uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    
#     # Button to clear history/reset
#     if st.button("Reset Interview"):
#         st.session_state.chat_history = []
#         st.session_state.vectorstore = None
#         st.experimental_rerun()

# # --- HELPER FUNCTIONS ---

# def process_pdf(uploaded_file):
#     """
#     1. Reads PDF
#     2. Splits text into chunks
#     3. Converts to Vectors (Embeddings)
#     4. Stores in FAISS
#     """
#     pdf_reader = PdfReader(uploaded_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
        
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
    
#     embeddings = OpenAIEmbeddings()
    
#     # CREATING THE FAISS VECTOR STORE
#     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
#     return vectorstore

# def generate_final_report(chat_history, role):
#     """
#     Takes the chat history and asks the LLM to grade the candidate.
#     """
#     llm = ChatOpenAI(model_name="gpt-4o")
    
#     # Format history for the prompt
#     conversation_text = ""
#     for msg in chat_history:
#         # Check if message is from user or AI
#         sender = "Candidate" if msg.type == 'human' else "Interviewer"
#         conversation_text += f"{sender}: {msg.content}\n"

#     report_prompt = f"""
#     You are a Senior Hiring Manager. Review the interview transcript below for the role of {role}.
    
#     TRANSCRIPT:
#     {conversation_text}
    
#     Generate a structured report in Markdown format:
#     1. **Technical Score** (1-10)
#     2. **Communication Score** (1-10)
#     3. **Strengths** (Bullet points)
#     4. **Weaknesses** (Bullet points)
#     5. **Final Hiring Recommendation** (Yes/No/Maybe)
#     """
    
#     response = llm.invoke(report_prompt)
#     return response.content

# # --- MAIN LOGIC ---

# # Initialize Session State
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None

# # 1. Process Resume
# if uploaded_file and st.session_state.vectorstore is None:
#     with st.spinner("Processing Resume and building FAISS Index..."):
#         st.session_state.vectorstore = process_pdf(uploaded_file)
#         st.success("Resume processed! You may start chatting.")

# # 2. Chat Interface
#     llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key)
#     #llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
    
#     # Context-aware prompt
#     template = f"""
#     You are a Technical Interviewer for a {job_role} role.
#     Use the Resume Context to ask specific questions.
    
#     Resume Context: {{context}}
#     Chat History: {{chat_history}}
#     Candidate Answer: {{question}}
    
#     Instructions:
#     1. Ask one question at a time.
#     2. Follow up on their previous answer.
#     3. Keep it professional.
#     """
    
#     QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

#     # The Chain
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=st.session_state.vectorstore.as_retriever(),
#         memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question', output_key='answer'),
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT}
#     )

#     # Display History
#     for msg in st.session_state.chat_history:
#         # Usually memory stores messages in a specific list, but we can also use the session state list if we sync them
#         # Here we rely on the memory buffer which updates automatically, but for display we can read the buffer:
#         role = "user" if msg.type == 'human' else "assistant"
#         with st.chat_message(role):
#             st.write(msg.content)

#     # User Input
#     if user_input := st.chat_input("Type your answer..."):
#         with st.chat_message("user"):
#             st.write(user_input)
        
#         # Get response
#         result = qa_chain.invoke({"question": user_input})
#         response_text = result["answer"]
        
#         # Update our manual history list for the report generator
#         # (Note: The memory buffer handles the chain, but we want a clean list for the report)
#         st.session_state.chat_history.append(result['chat_history'][-2]) # User msg
#         st.session_state.chat_history.append(result['chat_history'][-1]) # AI msg

#         with st.chat_message("assistant"):
#             st.write(response_text)

#     # --- GENERATE REPORT SECTION ---
#     st.divider()
#     if len(st.session_state.chat_history) > 2: # Only show if conversation has started
#         if st.button("📋 Generate Final Interview Report"):
#             with st.spinner("Grading candidate..."):
#                 report = generate_final_report(st.session_state.chat_history, job_role)
#                 st.markdown(report)
                
#                 # Optional: Save to file
#                 st.download_button("Download Report", report, file_name="interview_report.md")

# else:
#     st.info("Please upload a PDF resume to start.")