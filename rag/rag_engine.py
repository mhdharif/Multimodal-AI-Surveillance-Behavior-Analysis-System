import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.gemini_helper import generate_answer

CSV_PATH = "data/logs/alerts.csv"

def build_vector_db():

    df = pd.read_csv(CSV_PATH)

    docs = []

    for _, row in df.iterrows():

        text = f"""
        Timestamp: {row['Timestamp']}
        Person ID: {row['Person_ID']}
        Master ID: {row['Master_ID']}
        Event: {row['Event']}
        Snapshot: {row['Snapshot']}
        """

        docs.append(
            Document(page_content=text)
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    return db


def ask_rag(question):

    db = build_vector_db()

    results = db.similarity_search(
        question,
        k=5
    )

    retrieved_context = ""

    for doc in results:

        retrieved_context += doc.page_content
        retrieved_context += "\n\n"

    answer = generate_answer(
        question,
        retrieved_context
    )

    return answer, results