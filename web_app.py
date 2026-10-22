
import streamlit as st
import pymupdf
import numpy as np
import hashlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat


# 1. Configure the web page
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Professional UI styling
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
        color: #172B4D;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #102B46;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
        margin-bottom: 12px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    h1, h2, h3 {
        color: #102B46;
    }

    .hero {
        background: linear-gradient(
            110deg, #102B46, #176B83
        );
        padding: 28px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 32px;
    }

    .hero p {
        color: #D8E8F0;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# 3. Load embedding model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 4. Process PDF and create embeddings
@st.cache_resource(show_spinner=False)
def prepare_rag(pdf_bytes, pdf_name):

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunks = []
    chunk_size = 500

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        for start in range(0, len(text), chunk_size):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "page_number": page_number,
                    "source": pdf_name,
                    "text": chunk_text
                })

    total_pages = len(document)
    document.close()

    if not chunks:
        raise ValueError(
            "No readable text found in this PDF."
        )

    model = load_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    return chunks, embeddings, total_pages


# 5. Sidebar
with st.sidebar:

    st.title("🏦 DocuMind AI")
    st.caption("Document Intelligence")

    st.divider()

    st.subheader("📁 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload Your PDF Document",
        type=["pdf"]
    )

    st.caption(
        "Upload a text-based PDF to search its content."
    )

    st.divider()

    clear_chat = st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    )

    st.divider()
    st.caption("Powered by Ollama • Llama 3.2:3b")


# 6. Select document
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    pdf_name = uploaded_file.name
else:
    pdf_name = "banking.pdf"

    with open("data/banking.pdf", "rb") as file:
        pdf_bytes = file.read()

# 7. Track active document
document_id = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("document_id") != document_id:
    st.session_state.document_id = document_id
    st.session_state.messages = []

if clear_chat:
    st.session_state.messages = []


# 8. Prepare embeddings
try:
    with st.spinner("Preparing document knowledge base..."):
        chunks, embeddings, total_pages = prepare_rag(
            pdf_bytes,
            pdf_name
        )

except Exception as error:
    st.error(f"Document processing failed: {error}")
    st.stop()


# 9. Document statistics
with st.sidebar:
    st.subheader("📊 Document Overview")

    st.write(f"**Document:** {pdf_name}")
    st.write(f"**Pages:** {total_pages}")
    st.write(f"**Chunks:** {len(chunks)}")


# 10. Main heading
st.markdown("""
<div class="hero">
    <h1>Document AI Assistant</h1>
    <p>Search policies, understand rules, and get
    grounded answers powered by local AI.</p>
</div>
""", unsafe_allow_html=True)


st.caption(f"📄 Active Document: {pdf_name}")


# 11. Welcome message
if not st.session_state.messages:
    st.info(
        "Welcome! Ask a question about your uploaded "
        "banking policy document."
    )


# 12. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("pages"):
            st.caption(
                "Retrieved pages: "
                + ", ".join(
                    str(page)
                    for page in message["pages"]
                )
            )


# 13. Accept question
question = st.chat_input(
    "Ask anything about your document..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # 14. Generate question embedding
    model = load_model()
    question_embedding = model.encode([question])


    # 15. Semantic similarity
    similarities = cosine_similarity(
        question_embedding,
        embeddings
    )[0]


    # 16. Top 3 retrieval
    top_indices = np.argsort(similarities)[-3:][::-1]

    context = ""
    pages = []

    for index in top_indices:
        chunk = chunks[index]

        context += (
            f"\nPage {chunk['page_number']}:\n"
            f"{chunk['text']}\n"
        )

        pages.append(chunk["page_number"])


    # 17. Build grounded prompt
    prompt = f"""
You are a Document AI Assistant.

Use ONLY the provided document context.
Do not invent policies, numbers, or conditions.

If the answer is not in the context, say:
"Information not found in the document."

Include relevant page citations in the answer.

Document Context:
{context}

User Question:
{question}

Answer:
"""


    # 18. Generate response using Ollama
    with st.chat_message("assistant"):

        with st.spinner("Analyzing policy..."):

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

                unique_pages = sorted(set(pages))

                st.caption(
                    "Retrieved pages: "
                    + ", ".join(
                        str(page)
                        for page in unique_pages
                    )
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "pages": unique_pages
                })

            except Exception as error:
                st.error(f"Ollama error: {error}")
