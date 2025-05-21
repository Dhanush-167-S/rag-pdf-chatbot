import streamlit as st
import os
import uuid
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import langchain  

# Log LangChain version for debugging
st.write(f"LangChain version: {langchain.__version__}")

# Load environment variables
load_dotenv()

# Validate environment variables
if not os.getenv("HF_TOKEN") or not os.getenv("LANGCHAIN_API_KEY"):
    st.error("Missing HF_TOKEN or LANGCHAIN_API_KEY in .env file.")
    st.stop()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

# Set up Streamlit app
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload a PDF and chat with its content")

# Input the Groq API key
groq_api_key = st.text_input("Enter your Groq API key", type="password")

# Check if the API key is entered
if groq_api_key:
    try:
        llm = ChatGroq(model="Gemma2-9b-it", api_key=groq_api_key)

        
        session_id = st.text_input("Session ID", value="default_session")

        # File uploader
        uploaded_files = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)

        # Process uploaded files
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                # Use unique temporary file name
                temppdf = f"./temp_{uuid.uuid4()}.pdf"
                try:
                    with open(temppdf, "wb") as file:
                        file.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(temppdf)
                    docs = loader.load()
                    documents.extend(docs)
                finally:
                    if os.path.exists(temppdf):
                        os.remove(temppdf)  # Clean up temporary file

            # Split and embed the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever()

            # Contextualize question prompt
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference content in the chat history, "
                "formulate a standalone question that can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # Answer question prompt template
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n\n{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            user_input = st.chat_input("Enter your question here")
            if user_input:
                session_history = get_session_history(session_id)
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}},
                    )
                    with st.chat_message("assistant"):
                        st.write(response["answer"])

                    with st.expander("Session store"):
                        st.write(st.session_state.store)
                    
                    with st.expander("Chat History"):
                        for i,msg in enumerate(session_history.messages):
                            role = "user" if msg.type == "human" else "assistant"
                            with st.chat_message(role):
                                st.write(msg.content)
                            if msg.type == "assistant":
                                st.write("-"*100)
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

    except Exception as e:
        st.error(f"Invalid Groq API key or other error: {str(e)}")
else:
    st.warning("Please enter your Groq API key to proceed.")