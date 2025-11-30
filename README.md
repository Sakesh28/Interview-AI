##  Interview Agent

This​‍​‌‍​‍‌​‍​‌‍​‍‌ project is a RAG (Retrieval-Augmented Generation) system implementation aimed at automating technical candidate screening.
The platform enables a user to upload a resume (in PDF format), which is then parsed and indexed through FAISS (Facebook AI Similarity Search) and Hugging Face embeddings for fast semantic retrieval.
The principal intelligence is a Groq (with the Llama-3.1 model) that is the interviewer. So, by fetching the most relevant parts of the resume conversationally, 
the agent creates the next questions in the interview from the context and also evaluates the answers of the user on the spot.
The user interface is made with Streamlit to offer an uninterrupted, engaging chat ​‍​‌‍​‍‌​‍​‌‍​‍‌experience


Streamlit - https://interview-ai-4m6fuasqqfgwfxa5hh7vbz.streamlit.app/

### **Work-Flow**

[Candidate Resume (PDF)] 
        ⬇
[ 1. Ingestion Engine ]
(PyPDF extracts text -> Split into Chunks)
        ⬇
[ 2. Vector Embedding ]
(HuggingFace converts text to numbers -> Stored in FAISS DB)
       ⬇
[ 3. Interview Loop ] 
User Answer ➡ [Retrieval: Find Resume Context] ➡ [LLM Analysis (Groq/Llama-3)]
                                                        ⬇
                                            [AI Generates Next Question]
