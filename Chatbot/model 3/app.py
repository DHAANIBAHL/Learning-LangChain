
import streamlit as st
import pandas as pd
import pypdf
from docx import Document as DocxDocument
from chromadb.utils import embedding_functions
from groq import Groq

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-120b"

# DefaultEmbeddingFunction downloads a small local embedding model the first
# time it runs and uses it to turn text chunks into vectors. It is only used
# to compute embeddings here - nothing is stored in a vector database yet,
# and nothing is retrieved with it yet.
embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def extract_text(uploaded_file) -> str:
    """Read an uploaded file and return its plain-text content.
    Supports PDF, DOCX, TXT, CSV, and Excel files.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = DocxDocument(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return df.to_string(index=False)

    return ""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """Split text into overlapping word chunks.

    Chunking keeps each piece small enough to embed and search accurately,
    while the overlap avoids losing context that falls on a chunk boundary.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        # Move forward by (chunk_size - overlap) so consecutive chunks share
        # some words instead of cutting off mid-context
        i += chunk_size - overlap
    return chunks


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful personal assistant. Use the provided context from the user's documents when relevant, also you can answer general questions."}
    ]
if "doc_context" not in st.session_state:
    # Holds the raw extracted text of every uploaded file this session.
    # This is a simple stand-in for real retrieval - the whole document
    # gets stuffed into the prompt instead of just the relevant part.
    st.session_state.doc_context = {}
if "doc_chunks" not in st.session_state:
    # Holds the chunked text + embeddings for every uploaded file this
    # session. Nothing is stored in a vector database and nothing is
    # retrieved with these yet - they're just computed and kept here.
    st.session_state.doc_chunks = {}

with st.sidebar:
    st.title("Teach the bot")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Files are read and their text is added to the conversation context.",
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.doc_context:
                with st.spinner(f"Reading {f.name}..."):
                    text = extract_text(f)
                if text.strip():
                    st.session_state.doc_context[f.name] = text

                    # Split into overlapping chunks and compute an embedding
                    # for each one. These are kept in session state for now -
                    # not stored in a vector database and not used to answer
                    # questions yet.
                    chunks = chunk_text(text)
                    if chunks:
                        chunk_embeddings = embedding_fn(chunks)
                        st.session_state.doc_chunks[f.name] = list(
                            zip(chunks, chunk_embeddings)
                        )

                    st.success(f"Loaded {f.name}")
                else:
                    st.warning(f"Couldn't extract text from {f.name}")

    st.divider()

    if st.session_state.doc_context:
        with st.expander("Loaded files"):
            for fname in sorted(st.session_state.doc_context):
                st.write(f"- {fname}")

    if st.button("🗑️ Clear documents", use_container_width=True):
        st.session_state.doc_context = {}
        st.session_state.doc_chunks = {}
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Main chat
st.title("My Groq Chatbot")
st.caption("This is my personal Groq chatbot.")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build a separate list of messages just for the API call, so the raw
    # document text doesn't get permanently saved into the visible chat history
    api_messages = list(st.session_state.messages)
    if st.session_state.doc_context:
        context = "\n\n---\n\n".join(
            f"[Source: {name}]\n{text}" for name, text in st.session_state.doc_context.items()
        )
        api_messages.insert(
            -1,
            {
                "role": "system",
                "content": f"Context from the user's uploaded documents:\n\n{context}",
            },
        )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
