import os
import streamlit as st
from openai import RateLimitError

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate


# -----------------------------
# OpenAI API Key
# -----------------------------
os.environ["OPENAI_API_KEY"] = "YOUR_KEY"


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chat with your PDF's using OpenAI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Directory Based PDF Chatbot")
st.write("Ask questions from PDF documents stored inside a local directory.")


# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("Configuration")

pdf_path = st.sidebar.text_input(
    "Enter PDF Directory Path",
    value="./Udemy-Docs"
)

load_button = st.sidebar.button("Load PDF Documents")


# -----------------------------
# Initialize Session State
# -----------------------------
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "context" not in st.session_state:
    st.session_state.context = ""

if "total_documents" not in st.session_state:
    st.session_state.total_documents = 0


# -----------------------------
# Prompt Template
# -----------------------------
prompt = PromptTemplate.from_template("""
You are a helpful document-based chatbot.

Answer the user's question using only the context given below.
If the answer is not available in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")


# -----------------------------
# Load PDF Documents
# -----------------------------
if load_button:
    if not pdf_path:
        st.sidebar.error("Please enter a valid PDF directory path.")
    elif not os.path.exists(pdf_path):
        st.sidebar.error("Directory path does not exist.")
    else:
        try:
            with st.spinner("Loading PDF documents..."):
                loader = DirectoryLoader(
                    path=pdf_path,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )

                documents = loader.load()

                if len(documents) == 0:
                    st.warning("No PDF documents found in the given directory.")
                else:
                    context = "\n\n".join(doc.page_content for doc in documents)

                    st.session_state.context = context
                    st.session_state.total_documents = len(documents)
                    st.session_state.documents_loaded = True

                    st.success(
                        f"PDF documents loaded successfully. Total pages/documents loaded: {len(documents)}"
                    )

        except Exception as e:
            st.error(f"Error while loading documents: {e}")


# -----------------------------
# Main Chat Section
# -----------------------------
if st.session_state.documents_loaded:
    st.info(f"Total loaded PDF pages/documents: {st.session_state.total_documents}")

    question = st.text_input(
        "Ask your question:",
        value="What is tokenization, and how do tools like tiktoken and HuggingFace tokenizers help?"
    )

    ask_button = st.button("Ask")

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                chat = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_retries=3
                )

                formatted_prompt = prompt.format(
                    context=st.session_state.context,
                    question=question
                )

                st.subheader("Answer")

                answer_box = st.empty()
                full_answer = ""

                for chunk in chat.stream(formatted_prompt):
                    if chunk.content:
                        full_answer += chunk.content
                        answer_box.markdown(full_answer)

            except RateLimitError:
                st.error(
                    "Rate limit reached. Please wait and try again later, or check your OpenAI API usage limits."
                )

            except Exception as e:
                st.error(f"Error while generating answer: {e}")

else:
    st.warning("Please load PDF documents from the sidebar.")