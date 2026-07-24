
import streamlit as st
import tempfile
import cohere
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Task 1: Load and Chunk Document
def load_and_chunk_document(file_path, chunk_size=300, overlap=50):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only PDF and TXT files are supported.")
    
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(documents)

    return chunks

# Task 2: Create Embeddings
def create_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts)

    return embeddings

# Task 3: Semantic Search
def search_chunks(query, chunks, embeddings, top_k=3):
    query_embedding = embedding_model.encode([query])
    similarity = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarity.argsort()[-top_k:][::-1]
    results = [chunks[i].page_content for i in top_indices]
    return results

# Task 5: Cohere Answer Generation
def generate_answer(query, context, api_key):
    client = cohere.ClientV2(api_key=api_key)
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information given in the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content[0].text

# Streamlit UI
st.title("📄 RAG Document Question Answering")
st.write("Upload a TXT or PDF file and ask questions about it.")
api_key = st.text_input(
    "Enter Cohere API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Choose a TXT or PDF File",
    type=["txt", "pdf"]
)

query = st.text_input("Enter your Question")


if st.button("Search"):

    if api_key == "":
        st.warning("Please enter your Cohere API Key.")

    elif uploaded_file is None:
        st.warning("Please upload a document.")

    elif query.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="." + uploaded_file.name.split(".")[-1]
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())

            temp_path = temp_file.name

        with st.spinner("Processing..."):

            # Task 1
            chunks = load_and_chunk_document(temp_path)

            # Task 2
            embeddings = create_embeddings(chunks)

            # Task 3
            results = search_chunks(
                query,
                chunks,
                embeddings,
                top_k=3
            )

            # Create Context
            context = "\n\n".join(results)

            # Task 5
            answer = generate_answer(
                query,
                context,
                api_key
            )

        # Display Answer
        st.subheader("Generated Answer")

        st.success(answer)

        # Display Retrieved Chunks
        st.subheader("Retrieved Chunks")

        for i, chunk in enumerate(results, start=1):

            st.markdown(f"### Chunk {i}")

            st.write(chunk)

            st.divider()
