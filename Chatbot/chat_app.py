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


def retrieve_documents(query_embedding, top_k=5):
    distances, indices = faiss_index.search(
        np.array([query_embedding]).astype("float32"), k=top_k
    )
    retrieved_docs = []
    seen_sources = set()
    ref_counter = 1

    for d, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            doc = metadata[idx]
            source = doc.get("source", "")
            if source not in seen_sources:
                seen_sources.add(source)
                text = doc.get("extracted_text", "")
                doc["snippet"] = text[:300] + ("..." if len(text) > 300 else "")
                doc["distance"] = float(d)

                # Assign a reference ID so we can refer to it as [1], [2], etc.
                doc["ref_id"] = ref_counter
                ref_counter += 1

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

def make_inline_references_clickable(answer_text, docs):
    """
    Convert [1], [2], etc. in 'answer_text' into <a href="source" ...> links.
    Only do so if [n] is actually in the range of doc['ref_id'].
    """
    # Map ref_id -> doc for quick lookup
    ref_map = {doc["ref_id"]: doc for doc in docs}

    # Find all bracketed numbers, e.g. [1], [2], etc.
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, answer_text)

    # For each match, if it corresponds to a known doc, replace with a hyperlink
    for match in matches:
        ref_num = int(match)
        if ref_num in ref_map:
            source_url = ref_map[ref_num].get("source", "#")
            hyperlink = f'<a href="{source_url}" target="_blank">[{ref_num}]</a>'
            # Replace all occurrences of [ref_num] with the hyperlink
            answer_text = answer_text.replace(f"[{ref_num}]", hyperlink)

    return answer_text



@app.post("/query_form", response_class=HTMLResponse)
async def query_form(request: Request, query: str = Form(...), action: str = Form(...)):
    mode = "chat" if action == "augmented" else "generate" if action == "generated" else "chat"
    user_query = query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Retrieve docs
    query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
    retrieved_docs = retrieve_documents(query_embedding, top_k=5)

    # Combine their text
    combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
    max_tokens = 10000
    truncated_context = combined_context[:max_tokens]

    # If we are in generate mode, create a references note for GPT
    references_string = ""
    if mode == "generate":
        # e.g. "[1] No Intervention in Libya ( https://berniegrantarchive.org.uk/BGP52 )"
        for doc in retrieved_docs:
            rnum = doc["ref_id"]
            name = doc.get("Name", "Untitled")
            url = doc.get("source", "")
            references_string += f"[{rnum}] {name} ({url})\n"

    # Build the prompt
    if mode == "chat":
        prompt = (
            f"Context:\n{truncated_context}\n\n"
            f"Question: {user_query}\n\n"
            "Please answer the question using only the provided context. "
            "If the provided context does not contain sufficient information, "
            "please respond with: 'I'm sorry, but I don't have enough information on that. "
            "The answer generated is based on limited context and may not be entirely correct. "
            "Please use your discretion.'\n\nAnswer:"
        )
    elif mode == "generate":
        # We do NOT ask GPT to produce a references block at the end; only inline
        # prompt = (
        #     f"Context:\n{truncated_context}\n\n"
        #     "If you use info from any document, please cite it inline in brackets "
        #     "matching the reference IDs below:\n\n"
        #     f"{references_string}\n"
        #     f"Question: {user_query}\n\n"
        #     "Please provide your answer with any inline citations as needed. "
        #     "Please use HTML tags for formatting rather than Markdown"
        #     "Please provide an answer based on the provided context. "
        #     "If the context is limited, you may expand while ensuring accuracy. Accuracy is very important "
        #     "Be very formal, polite, respectful and politically correct"
        #     "If the question cannot be answered accurately, it's better to say so.\n\nAnswer:"
        # )
        prompt = (
            f"Context:\n{truncated_context}\n\n"
            "If you reference information from any document, please cite it inline using brackets "
            "corresponding to the provided reference IDs below:\n\n"
            f"{references_string}\n\n"
            f"Question: {user_query}\n\n"
            "Please provide your answer with any inline citations as needed. "
            "Please provide a multilayered and insightful response, ensuring you thoroughly address the question. "
            "Incorporate thoughtful analysis, nuance, and contextually relevant elaboration where appropriate. "
            "Use HTML tags exclusively for formatting (do not use Markdown). "
            "Prioritize accuracy above all, grounding your answer firmly in the provided context; however, "
            "if necessary due to context limitations, carefully expand your explanation while maintaining factual accuracy. "
            "Maintain a consistently formal, respectful, polite, and politically correct tone throughout. "
            "If the provided context does not allow for an accurate or insightful response, explicitly state so clearly.\n\n"
            "Answer:"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode.")

    # Call OpenAI
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

    # Post-process the answer to turn [1], [2], etc. into clickable links
    answer = make_inline_references_clickable(answer, retrieved_docs)

    # Prepare related info
    related_information = []
    for doc in retrieved_docs:
        images = get_images(doc)
        snippet_text = doc.get("snippet", "")
        snippet_text_cleaned = re.sub(r'<.*?>', '', snippet_text)
        related_information.append({
            "Name": doc.get("Name", ""),
            "snippet": snippet_text_cleaned,
            "source": doc.get("source", ""),
            "search_source": doc.get("search_source", ""),
            "images": images,
            "ref_id": doc["ref_id"]  # for labeling in HTML
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": answer,
        "related_information": related_information,
        "query": user_query,
        "dark_mode": False
    })

