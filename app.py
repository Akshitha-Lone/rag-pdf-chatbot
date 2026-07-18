%pip install -q gradio chromadb pypdf
import gradio as gr
import chromadb
import google.generativeai as genai
import uuid
from pypdf import PdfReader # Corrected PdfReader capitalization
from sentence_transformers import SentenceTransformer

# Gemini API - Replace "" with your actual API key
genai.configure(api_key="")
llm=genai.GenerativeModel("gemini-2.5-flash")

# Embedding Model
print("Loading embedding model")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("rag_docs")

# Extract PDF Text
def extract_text(pdf):
  reader = PdfReader(pdf.name)
  text = ""
  for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
      text += page_text + "\n"
  return text

# Chunk text (moved to top level)
def chunk_text(text, size=300):
  return [text[i:i+size] for i in range(0,len(text), size)]

# Upload PDF
def upload_pdf(pdf):
  global collection
  text = extract_text(pdf)
  chunks = chunk_text(text)
  print(f"chunks Created: {len(chunks)}")
  embeddings = embedder.encode(chunks).tolist()

  # Reset collection for new pdf
  try:
    client.delete_collection("rag_docs")
  except:
    pass
  collection = client.get_or_create_collection("rag_docs")
  collection.add(
      documents=chunks,
      embeddings=embeddings,
      ids=[str(uuid.uuid4()) for _ in chunks]
  )
  return f"PDF Indexed Successfully ({len(chunks)} chunks)"

# Ask question
def ask(question):
  query_embeddings = embedder.encode(question).tolist() # Added assignment operator
  result = collection.query(
      query_embeddings = [query_embeddings],
      n_results = 3
  )
  context = "\n".join(result["documents"][0])
  context = context[:4000]  #prevent timeout

  prompt =f"""
  Answer only using the context below.
  If the answer is not found, say:
  'iI couldn't find the answer in the document.'
  context:
  {context}
  Question
  {question}
  """

  response = llm.generate_content(prompt) # Corrected indentation
  return response.text # Corrected indentation

with gr.Blocks() as demo:
    gr.Markdown("# 📄 RAG PDF Chatbot")

    pdf = gr.File(label="Upload PDF")
    upload_btn = gr.Button("Index PDF")
    status = gr.Textbox(label="Status")

    upload_btn.click(upload_pdf, inputs=pdf, outputs=status)

    question = gr.Textbox(label="Ask a Question")
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=10)

    ask_btn.click(ask, inputs=question, outputs=answer)

demo.launch(debug=True)
