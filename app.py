
import pymupdf

# Step 1: Open PDF
pdf_path = "data/banking.pdf"
document = pymupdf.open(pdf_path)

# Step 2: Create empty list
chunks = []

# Step 3: Define chunk size
chunk_size = 500

# Step 4: Read each page
for page_number, page in enumerate(document, start=1):

    text = page.get_text()

    # Step 5: Divide text into chunks
    for start in range(0, len(text), chunk_size):

        chunk_text = text[start:start + chunk_size]

        # Step 6: Add metadata
        chunk = {
            "chunk_id": len(chunks) + 1,
            "page_number": page_number,
            "source": pdf_path,
            "text": chunk_text
        }

        chunks.append(chunk)

document.close()

# Step 7: Display results
print("Total chunks created:", len(chunks))

print("\nFirst Chunk:")
print(chunks[0])


# Step 8: Embeddings and Semantic Search

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get text from all chunks
chunk_texts = [chunk["text"] for chunk in chunks]

# Convert chunks into vectors
embeddings = model.encode(chunk_texts)

print("Embeddings Shape:", embeddings.shape)

# User question
question = "What is the minimum monthly income required for a personal loan?"

# Convert question into vector
question_embedding = model.encode([question])

# Calculate similarity
similarities = cosine_similarity(
    question_embedding,
    embeddings
)[0]

# Find best matching chunk
best_chunk_index = similarities.argmax()

print("\nQuestion:", question)
print("\nRelevant Banking Policy:")
print(chunks[best_chunk_index]["text"])

print("\nPage Number:",
      chunks[best_chunk_index]["page_number"])


# Step 8: Top 3 Retrieval

import numpy as np

top_3_indices = np.argsort(similarities)[-3:][::-1]

print("\nTOP 3 RELEVANT CHUNKS")

for index in top_3_indices:
    print("\n----------------------")
    print("Chunk ID:", chunks[index]["chunk_id"])
    print("Page:", chunks[index]["page_number"])
    print("Score:", round(float(similarities[index]), 4))
    print("Text:", chunks[index]["text"])


# Step 10: Connect Banking RAG with Ollama

from ollama import chat

# Collect top 3 relevant chunks
context = ""

for index in top_3_indices:
    page = chunks[index]["page_number"]
    text = chunks[index]["text"]

    context += f"\nPage {page}:\n{text}\n"

# Prepare instructions for the LLM
prompt = f"""
You are a Banking Policy Assistant.

Answer the question using ONLY the policy context.

Rules:
1. Do not invent information.
2. Give a clear answer.
3. Include the source page number.
4. If the answer is not available in the context,
   say "Information not found in the policy."

Policy Context:
{context}

Question:
{question}

Answer:
"""

# Send prompt to local Ollama model
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

# Print final generated answer
print("\nFINAL BANKING RAG ANSWER:")
print(response.message.content)
