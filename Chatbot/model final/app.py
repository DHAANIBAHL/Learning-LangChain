
import os
import time
import uuid
import hashlib

import streamlit as st
import pandas as pd
import pypdf
from docx import Document as DocxDocument
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

st.set_page_config(
    page_title="Personal Groq Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-120b"

# Vector database setup 
# CHROMA_DIR is where the vector database is saved on disk, so uploaded
# knowledge persists even after the app restarts.
CHROMA_DIR = "./chroma_db"

# DefaultEmbeddingFunction downloads a small local embedding model the first time it runs and uses it to turn text into vectors.
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# PersistentClient writes the vector database to CHROMA_DIR instead of keeping it only in memory.
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# get_or_create_collection either loads the existing "personal_knowledge" collection from disk, or creates a new empty one the first time.
collection = chroma_client.get_or_create_collection(
    name="personal_knowledge",
    embedding_function=embedding_fn,
)

# Number of most-relevant chunks to retrieve per question
TOP_K = 4


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


def index_file(uploaded_file):
    """Extract, chunk, and store a file's content in the vector database.

    Returns the number of chunks stored (0 if nothing could be extracted).
    """
    text = extract_text(uploaded_file)
    if not text.strip():
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Hash the file contents so re-uploading the exact same file doesn't create duplicate entries with clashing IDs
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
    ids = [f"{uploaded_file.name}-{file_hash}-{i}" for i in range(len(chunks))]

    # Store which file each piece of text came from, so answers can be traced back to a source later if needed
    metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

    # Add everything to the vector database in one call. Chroma automatically embeds each chunk using embedding_fn.
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def retrieve_context(query: str, k: int = TOP_K) -> str:
    """Search the vector database for the chunks most relevant to a query.

    Returns a formatted string of the top matches, or an empty string if
    the knowledge base is empty / nothing relevant is found.
    """
    if collection.count() == 0:
        return ""

    # Chroma embeds the query text and finds the nearest stored chunks
    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return ""

    # Format each chunk with its source filename for traceability
    parts = []
    for doc, meta in zip(docs, metas):
        parts.append(f"[Source: {meta.get('source', 'unknown')}]\n{doc}")
    return "\n\n---\n\n".join(parts)


# Session state 
# st.session_state persists values across Streamlit reruns (every user interaction reruns the whole script from top to bottom)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful personal assistant. Use the provided context from the user's documents when relevant, also you can answer general questions."}
    ]
if "indexed_files" not in st.session_state:
    # Tracks which filenames have already been indexed this session, so re-running the script doesn't re-index the same file repeatedly
    st.session_state.indexed_files = set()

with st.sidebar:
    st.title("Teach the bot")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Files are chunked, embedded, and stored locally so the bot can search them.",
    )

    if uploaded_files:
        for f in uploaded_files:
            # Only index files that haven't already been processed this session
            if f.name not in st.session_state.indexed_files:
                with st.spinner(f"Indexing {f.name}..."):
                    n_chunks = index_file(f)
                if n_chunks > 0:
                    st.session_state.indexed_files.add(f.name)
                    st.success(f"Indexed {f.name} ({n_chunks} chunks)")
                else:
                    st.warning(f"Couldn't extract text from {f.name}")

    st.divider()

    # Show which files are currently indexed
    if st.session_state.indexed_files:
        with st.expander("Indexed files"):
            for fname in sorted(st.session_state.indexed_files):
                st.write(f"- {fname}")

    # Wipe the entire vector database and start fresh
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        chroma_client.delete_collection("personal_knowledge")
        collection = chroma_client.get_or_create_collection(
            name="personal_knowledge", embedding_function=embedding_fn
        )
        st.session_state.indexed_files = set()
        # restart the script so the UI reflects the cleared state
        st.rerun()  

    st.divider()

    # Reset the visible chat history but keep the knowledge base intact
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]  # keep only the system prompt
        st.rerun()

# Main chat 
st.title("My Groq Chatbot")
st.caption("Upload your files in the sidebar, then ask questions about them — or anything else.")

# Redraw the full conversation history on every rerun while skipping the system prompt
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    # Save and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Look up relevant chunks from uploaded documents for this question
    context = retrieve_context(prompt)

    # Build a separate list of messages just for the API call, so the retrieved context doesn't get permanently saved into the visible chat history
    api_messages = list(st.session_state.messages)
    if context:
        api_messages.insert(
            -1,  # insert just before the latest user message
            {
                "role": "system",
                "content": f"Relevant context from the user's uploaded documents:\n\n{context}",
            },
        )

    # Stream the assistant's reply
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
            # "▌" acts as a blinking cursor while the response is still typing
            placeholder.markdown(full_response + "▌")
            time.sleep(0.02)  

        placeholder.markdown(full_response)  

    # Save the completed assistant reply into permanent chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
