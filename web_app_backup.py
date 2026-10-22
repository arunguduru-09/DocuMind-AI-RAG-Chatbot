
import streamlit as st
import pymupdf
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat


# Step 1: Page configuration
st.set_page_config(
    page_title="Banking RAG Chatbot",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Banking Policy AI Assistant")

st.write("Ask questions about your Banking Policy PDF.")


# Step 2: Load PDF and create embeddings
@st.cache_resource
def prepare_rag():

    pdf_path = "data/banking.pdf"

    document = pymupdf.open(pdf_path)

    chunks = []
    chunk_size = 500

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        for start in range(0, len(text), chunk_size):

            chunk_text = text[start:start + chunk_size]

            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_number": page_number,
                "source": pdf_path,
                "text": chunk_text
            })

    document.close()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(chunk_texts)

    return chunks, model, embeddings


# Step 3: Initialize RAG
with st.spinner("Loading Banking Policy..."):
    chunks, model, embeddings = prepare_rag()


# Step 4: Chat history
if "messages" not in st.session_state:

    st.session_state.messages = []


# Step 5: Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Step 6: Accept user question
question = st.chat_input("Ask a banking policy question...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # Step 7: Convert question into embedding
    question_embedding = model.encode([question])


    # Step 8: Calculate similarity
    similarities = cosine_similarity(
        question_embedding,
        embeddings
    )[0]


    # Step 9: Retrieve top 3 chunks
    top_3_indices = np.argsort(similarities)[-3:][::-1]


    # Step 10: Prepare context
    context = ""

    source_pages = []

    for index in top_3_indices:

        page = chunks[index]["page_number"]
        text = chunks[index]["text"]

        context += f"\nPage {page}:\n{text}\n"

        source_pages.append(page)


    # Step 11: Create prompt
    prompt = f"""
You are a Banking Policy Assistant.

Answer the question using ONLY the provided policy context.

Rules:
1. Do not invent information.
2. Give a simple, clear answer.
3. Mention the source page number.
4. If the answer is not in the context,
   say "Information not found in the policy."

Policy Context:
{context}

Question:
{question}

Answer:
"""


    # Step 12: Call Ollama LLM
    with st.chat_message("assistant"):

        with st.spinner("Searching Banking Policy..."):

            try:
                response = chat(
                    model="llama3.2:3b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={"temperature": 0}
                )

                answer = response.message.content

                st.markdown(answer)

                st.caption(
                    "Retrieved pages: "
                    + ", ".join(
                        str(page)
                        for page in sorted(set(source_pages))
                    )
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:
                st.error(
                    "Unable to connect to Ollama: "
                    + str(error)
                )
