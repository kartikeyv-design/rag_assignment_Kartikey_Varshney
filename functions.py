
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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

    return splitter.split_documents(documents)


def create_embeddings(chunks):

    texts = [chunk.page_content for chunk in chunks]

    return embedding_model.encode(texts)


def search_chunks(query, chunks, embeddings, top_k=3):

    query_embedding = embedding_model.encode([query])

    similarity = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarity.argsort()[-top_k:][::-1]

    return [
        chunks[i].page_content
        for i in top_indices
    ]


def generate_answer(query, context, api_key):

    client = cohere.ClientV2(api_key=api_key)

    prompt = f"""
    Answer only using the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.message.content[0].text
