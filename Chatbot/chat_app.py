from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv
import re

app = FastAPI()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")

base_folder = os.path.dirname(os.getcwd())
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai_index_folder = os.path.join(base_folder, "models", "openAI")
faiss_index_file = os.path.join(openai_index_folder, "combined_index.faiss")
metadata_file = os.path.join(openai_index_folder, "combined_metadata.json")

faiss_index = faiss.read_index(faiss_index_file)
with open(metadata_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def get_embedding(text, model_name="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model_name)
    return response["data"][0]["embedding"]


def retrieve_documents(query_embedding, top_k=10):
    distances, indices = faiss_index.search(np.array([query_embedding]).astype("float32"), k=top_k)
    retrieved_docs = []
    seen_sources = set()

    for d, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            doc = metadata[idx]
            source = doc.get("source", "")
            if source not in seen_sources:
                seen_sources.add(source)
                text = doc.get("extracted_text", "")
                doc["snippet"] = text[:300] + ("..." if len(text) > 300 else "")
                doc["distance"] = float(d)
                retrieved_docs.append(doc)
    return retrieved_docs


def get_images(doc):
    """
    Returns a list of image filenames or paths from doc['png_info'] if present.
    """
    png_info = doc.get("png_info", None)
    if isinstance(png_info, list):
        return png_info if png_info else []
    if isinstance(png_info, str) and png_info.strip():
        return [png_info]
    return []


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "dark_mode": False  # Default to light mode
    })


@app.post("/query_form", response_class=HTMLResponse)
async def query_form(request: Request, query: str = Form(...), action: str = Form(...)):
    mode = "chat" if action == "augmented" else "generate" if action == "generated" else "chat"
    user_query = query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    print("query: "+user_query+"\n")
    # Retrieve documents
    query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
    retrieved_docs = retrieve_documents(query_embedding, top_k=5)

    # Build context for AI response
    combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
    max_tokens = 10000
    truncated_context = combined_context[:max_tokens]

    # Build the prompt depending on mode
    if mode == "chat":
        prompt = (
            f"Context:\n{truncated_context}\n\n"
            f"Question: {user_query}\n\n"
            "Please answer the question using only the provided context. "
            "If the provided context does not contain sufficient information, "
            "please respond with: 'I'm sorry, but I don't have enough information on that. "
            "The answer generated is based on limited context and may not be entirely correct. Please use your discretion.' "
            "\n\nAnswer:"
        )
    elif mode == "generate":
        prompt = (
            f"Context:\n{truncated_context}\n\n"
            f"Question: {user_query}\n\n"
            "Please provide an answer based on the provided context. "
            "If the context is limited, you may expand upon it while ensuring accuracy. "
            "Be respectful and considerate when generating responses. "
            "If the question cannot be answered with certainty, it's better not to answer than to provide misleading or incorrect information. "
            "\n\nAnswer:"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'chat' or 'generate'.")

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for the Bernie Grant digital archive."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # Prepare related information (with images bound to their respective docs)
    related_information = []
    for doc in retrieved_docs:
        images = get_images(doc)
        snippet_text = doc.get("snippet", "")
        snippet_text_cleaned = re.sub(r'<.*?>', '', snippet_text)  # Remove any HTML tags
        related_information.append({
            "Name": doc.get("Name", ""),
            "snippet": snippet_text_cleaned,
            "source": doc.get("source", ""),
            "search_source": doc.get("search_source", ""),
            "images": images
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": answer,
        "related_information": related_information,
        "query": user_query,
        "dark_mode": False
    })
