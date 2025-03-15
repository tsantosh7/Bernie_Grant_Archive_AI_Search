# import json
# import faiss
# import numpy as np
import re
# from fastapi import FastAPI, HTTPException, Request, Form
# import openai
# import tiktoken
# import os
# from pathlib import Path
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from dotenv import load_dotenv
#
# app = FastAPI()
# load_dotenv()  # Loads variables from .env into the environment
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# if openai.api_key is None:
#     print("API key not found. Please set the OPENAI_API_KEY environment variable.")
# else:
#     print("API key found:", openai.api_key)
#
# # Set base_folder to the parent folder of the current working directory using os.path.dirname
# base_folder = os.path.dirname(os.getcwd())
#
# # Mount static files (e.g., logo.png)
# app.mount("/static", StaticFiles(directory="static"), name="static")
#
# # Initialize Jinja2 templates (ensure you have a "templates" folder)
# templates = Jinja2Templates(directory="templates")
#
# # Load tokenizer for GPT-3.5-turbo
# tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
#
# # Paths for combined FAISS index and metadata (generated from your indexing code)
# openai_index_folder = os.path.join(base_folder, "models", "openAI")
# faiss_index_file = os.path.join(openai_index_folder, "combined_index.faiss")
# metadata_file = os.path.join(openai_index_folder, "combined_metadata.json")
#
# # Load the combined FAISS index and metadata.
# faiss_index = faiss.read_index(faiss_index_file)
# with open(metadata_file, "r", encoding="utf-8") as f:
#     metadata = json.load(f)
#
# print("Combined FAISS index loaded successfully!")
#
# def get_embedding(text, model_name="text-embedding-ada-002"):
#     # Uncomment below for real embeddings:
#     response = openai.Embedding.create(
#         input=[text],
#         model=model_name
#     )
#     return response["data"][0]["embedding"]
#     # return np.random.rand(1536).tolist()
#
# def retrieve_documents(query_embedding, top_k=10):
#     distances, indices = faiss_index.search(np.array([query_embedding]).astype("float32"), k=top_k)
#     retrieved_docs = []
#     for d, idx in zip(distances[0], indices[0]):
#         if idx < len(metadata):
#             doc = metadata[idx]
#             text = doc.get("extracted_text", "")
#             doc["snippet"] = text[:300] + ("..." if len(text) > 300 else "")
#             doc["distance"] = float(d)
#             retrieved_docs.append(doc)
#     return retrieved_docs
#
# def compute_source_link(doc):
#     """
#     Returns the 'search_source' from the document metadata if available,
#     otherwise, falls back to a URL built from the source field.
#     """
#     if doc.get("search_source", "").strip():
#         return doc["search_source"]
#     src = doc.get("source", "")
#     if src:
#         return f"https://berniegrantarchive.org.uk/?s={src}"
#     return ""
#
# def get_top_image(png_info):
#     """If png_info is a list, return the first element; if it's a string, return it; otherwise, return None."""
#     if isinstance(png_info, list):
#         return png_info[0] if png_info else None
#     if isinstance(png_info, str) and png_info.strip():
#         return png_info
#     return None
#
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "answer": None,
#         "evidence": None,
#         "source_link": None,
#         "related_information": [],
#         "query": ""
#     })
#
# @app.post("/query_form", response_class=HTMLResponse)
# async def query_form(
#         request: Request,
#         query: str = Form(...),
#         action: str = Form(...)
# ):
#     mode = "chat" if action == "augmented" else "generate" if action == "generated" else "chat"
#     user_query = query.strip()
#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")
#
#     # Retrieve documents from the combined index.
#     query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
#     retrieved_docs = retrieve_documents(query_embedding, top_k=10)
#
#     # Combine retrieved texts into context.
#     combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
#     max_tokens = 10000
#     context_tokens = tokenizer.encode(combined_context)
#     truncated_context = tokenizer.decode(context_tokens[:max_tokens])
#
#     # Build prompt based on mode.
#     if mode == "chat":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please answer the question using only the provided context. "
#             "If the provided context does not contain sufficient information, "
#             "please respond with: \"I'm sorry, but I don't have enough information on that. "
#             "The answer generated is based on limited context and may not be entirely correct. Please use your discretion.\" \n\n"
#             "Answer:"
#         )
#     else:
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please provide an answer based on the provided context. "
#             "If the context is limited, feel free to generate additional content while ensuring the answer remains relevant. \n\n"
#             "Answer:"
#         )
#
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI assistant for the Bernie Grant digital archive."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         answer = response["choices"][0]["message"]["content"]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
#
#     # Determine evidence and source link using the top retrieved document.
#     threshold = 0.9  # adjust threshold as needed
#     evidence = None
#     source_link = None
#     if retrieved_docs:
#         first_doc = retrieved_docs[0]
#         if first_doc.get("distance", 1.0) < threshold:
#             evidence = get_top_image(first_doc.get("png_info"))
#             source_link = compute_source_link(first_doc)
#
#     # Build related information for top 5 documents.
#     related_information = []
#     for doc in retrieved_docs[:5]:
#         related_information.append({
#             "Name": doc.get("Name", ""),
#             "snippet": doc.get("snippet", ""),
#             "source": doc.get("source", ""),
#             "search_source": doc.get("search_source", ""),
#             "distance": doc.get("distance", 0)
#         })
#
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "answer": answer,
#         "evidence": evidence,
#         "source_link": source_link,
#         "related_information": related_information,
#         "query": user_query
#     })
#
# @app.post("/query")
# async def query_openai(payload: dict):
#     user_query = payload.get("query", "").strip()
#     mode = payload.get("mode", "chat").strip().lower()
#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")
#
#     query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
#     retrieved_docs = retrieve_documents(query_embedding, top_k=10)
#     combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
#     max_tokens = 10000
#     context_tokens = tokenizer.encode(combined_context)
#     truncated_context = tokenizer.decode(context_tokens[:max_tokens])
#
#     if mode == "chat":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please answer the question using only the provided context. "
#             "If the provided context does not contain sufficient information, "
#             "please respond with: \"I'm sorry, but I don't have enough information on that. "
#             "The answer generated is based on limited context and may not be entirely correct. Please use your discretion.\" \n\n"
#             "Answer:"
#         )
#     elif mode == "generate":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please provide an answer based on the provided context. "
#             "If the context is limited, feel free to generate additional content while ensuring the answer remains relevant. \n\n"
#             "Answer:"
#         )
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode. Choose 'chat' or 'generate'.")
#
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI assistant for the Bernie Grant digital archive."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         answer = response["choices"][0]["message"]["content"]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
#
#     threshold = 0.5
#     evidence = None
#     if retrieved_docs:
#         first_doc = retrieved_docs[0]
#         if first_doc.get("distance", 1.0) < threshold:
#             evidence = get_top_image(first_doc.get("png_info"))
#
#     return {
#         "answer": answer,
#         "retrieved_documents": retrieved_docs,
#         "evidence": evidence,
#     }

############################################################################################
# from fastapi import FastAPI, HTTPException, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import json
# import faiss
# import numpy as np
# import openai
# import os
# from dotenv import load_dotenv
# import re
#
# app = FastAPI()
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# if openai.api_key is None:
#     print("API key not found. Please set the OPENAI_API_KEY environment variable.")
#
# base_folder = os.path.dirname(os.getcwd())
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# openai_index_folder = os.path.join(base_folder, "models", "openAI")
# faiss_index_file = os.path.join(openai_index_folder, "combined_index.faiss")
# metadata_file = os.path.join(openai_index_folder, "combined_metadata.json")
#
# faiss_index = faiss.read_index(faiss_index_file)
# with open(metadata_file, "r", encoding="utf-8") as f:
#     metadata = json.load(f)
#
#
# def get_embedding(text, model_name="text-embedding-ada-002"):
#     response = openai.Embedding.create(input=[text], model=model_name)
#     return response["data"][0]["embedding"]
#
#
# def retrieve_documents(query_embedding, top_k=10):
#     distances, indices = faiss_index.search(np.array([query_embedding]).astype("float32"), k=top_k)
#     retrieved_docs = []
#     seen_sources = set()
#
#     for d, idx in zip(distances[0], indices[0]):
#         if idx < len(metadata):
#             doc = metadata[idx]
#             source = doc.get("source", "")
#             if source not in seen_sources:
#                 seen_sources.add(source)
#                 text = doc.get("extracted_text", "")
#                 doc["snippet"] = text[:300] + ("..." if len(text) > 300 else "")
#                 doc["distance"] = float(d)
#                 retrieved_docs.append(doc)
#     return retrieved_docs
#
#
# def compute_source_link(doc):
#     if doc.get("search_source", "").strip():
#         return doc["search_source"]
#     return doc.get("source", "")
#
#
# def get_images(doc):
#     png_info = doc.get("png_info", None)
#     if isinstance(png_info, list):
#         return png_info if png_info else []
#     if isinstance(png_info, str) and png_info.strip():
#         return [png_info]
#     return []
#
#
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
#
#
# @app.post("/query_form", response_class=HTMLResponse)
# async def query_form(request: Request, query: str = Form(...), action: str = Form(...)):
#     mode = "chat" if action == "augmented" else "generate" if action == "generated" else "chat"
#     user_query = query.strip()
#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")
#
#     query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
#     retrieved_docs = retrieve_documents(query_embedding, top_k=5)
#
#     # Build context for AI response
#     combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
#     max_tokens = 10000
#     truncated_context = combined_context[:max_tokens]
#
#     if mode == "chat":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please answer the question using only the provided context. "
#             "If the provided context does not contain sufficient information, "
#             "please respond with: 'I'm sorry, but I don't have enough information on that. Please try 'AI generated Retrieval' "
#             # "The answer generated is based on limited context and may not be entirely correct. Please use your discretion.' "
#             "\n\nAnswer:"
#         )
#     elif mode == "generate":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please provide an answer based on the provided context. "
#             "If the context is limited, you may expand upon it while ensuring accuracy. "
#             "Be respectful and considerate when generating responses. "
#             "If the question cannot be answered with certainty, it's better not to answer than to provide misleading or incorrect information. "
#             "\n\nAnswer:"
#         )
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode. Choose 'chat' or 'generate'.")
#
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI assistant for the Bernie Grant digital archive."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         answer = response["choices"][0]["message"]["content"]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
#
#     evidence_images = []
#     related_information = []
#     for doc in retrieved_docs:
#         images = get_images(doc)
#         evidence_images.extend(images)
#         snippet_text = doc.get("snippet", "")
#         snippet_text_cleaned = re.sub(r'<.*?>', '', snippet_text)  # Remove HTML tags
#         related_information.append({
#             "Name": doc.get("Name", ""),
#             "snippet": snippet_text_cleaned,
#             "source": doc.get("source", ""),
#             "search_source": doc.get("search_source", "")
#         })
#
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "answer": answer,
#         "evidence": evidence_images,
#         "related_information": related_information,
#         "query": user_query
#     })
#


#################################################################################

# from fastapi import FastAPI, HTTPException, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import json
# import faiss
# import numpy as np
# import openai
# import os
# from dotenv import load_dotenv
# import re
#
# app = FastAPI()
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# if openai.api_key is None:
#     print("API key not found. Please set the OPENAI_API_KEY environment variable.")
#
# base_folder = os.path.dirname(os.getcwd())
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# openai_index_folder = os.path.join(base_folder, "models", "openAI")
# faiss_index_file = os.path.join(openai_index_folder, "combined_index.faiss")
# metadata_file = os.path.join(openai_index_folder, "combined_metadata.json")
#
# faiss_index = faiss.read_index(faiss_index_file)
# with open(metadata_file, "r", encoding="utf-8") as f:
#     metadata = json.load(f)
#
#
# def get_embedding(text, model_name="text-embedding-ada-002"):
#     response = openai.Embedding.create(input=[text], model=model_name)
#     return response["data"][0]["embedding"]
#
#
# def retrieve_documents(query_embedding, top_k=10):
#     distances, indices = faiss_index.search(np.array([query_embedding]).astype("float32"), k=top_k)
#     retrieved_docs = []
#     seen_sources = set()
#
#     for d, idx in zip(distances[0], indices[0]):
#         if idx < len(metadata):
#             doc = metadata[idx]
#             source = doc.get("source", "")
#             if source not in seen_sources:
#                 seen_sources.add(source)
#                 text = doc.get("extracted_text", "")
#                 doc["snippet"] = text[:300] + ("..." if len(text) > 300 else "")
#                 doc["distance"] = float(d)
#                 retrieved_docs.append(doc)
#     return retrieved_docs
#
#
# def compute_source_link(doc):
#     if doc.get("search_source", "").strip():
#         return doc["search_source"]
#     return doc.get("source", "")
#
#
# def get_images(doc):
#     png_info = doc.get("png_info", None)
#     if isinstance(png_info, list):
#         return png_info if png_info else []
#     if isinstance(png_info, str) and png_info.strip():
#         return [png_info]
#     return []
#
#
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "dark_mode": False  # Default to light mode
#     })
#
#
# @app.post("/query_form", response_class=HTMLResponse)
# async def query_form(request: Request, query: str = Form(...), action: str = Form(...)):
#     mode = "chat" if action == "augmented" else "generate" if action == "generated" else "chat"
#     user_query = query.strip()
#     if not user_query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")
#
#     query_embedding = get_embedding(user_query, model_name="text-embedding-ada-002")
#     retrieved_docs = retrieve_documents(query_embedding, top_k=5)
#
#     # Build context for AI response
#     combined_context = "\n\n".join([doc.get("extracted_text", "") for doc in retrieved_docs])
#     max_tokens = 10000
#     truncated_context = combined_context[:max_tokens]
#
#     if mode == "chat":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please answer the question using only the provided context. "
#             "If the provided context does not contain sufficient information, "
#             "please respond with: 'I'm sorry, but I don't have enough information on that. "
#             "The answer generated is based on limited context and may not be entirely correct. Please use your discretion.' "
#             "\n\nAnswer:"
#         )
#     elif mode == "generate":
#         prompt = (
#             f"Context:\n{truncated_context}\n\n"
#             f"Question: {user_query}\n\n"
#             "Please provide an answer based on the provided context. "
#             "If the context is limited, you may expand upon it while ensuring accuracy. "
#             "Be respectful and considerate when generating responses. "
#             "If the question cannot be answered with certainty, it's better not to answer than to provide misleading or incorrect information. "
#             "\n\nAnswer:"
#         )
#     else:
#         raise HTTPException(status_code=400, detail="Invalid mode. Choose 'chat' or 'generate'.")
#
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI assistant for the Bernie Grant digital archive."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         answer = response["choices"][0]["message"]["content"]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
#
#     evidence_images = []
#     related_information = []
#     for doc in retrieved_docs:
#         images = get_images(doc)
#         evidence_images.extend(images)
#         snippet_text = doc.get("snippet", "")
#         snippet_text_cleaned = re.sub(r'<.*?>', '', snippet_text)  # Remove HTML tags
#         related_information.append({
#             "Name": doc.get("Name", ""),
#             "snippet": snippet_text_cleaned,
#             "source": doc.get("source", ""),
#             "search_source": doc.get("search_source", "")
#         })
#
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "answer": answer,
#         "evidence": evidence_images,
#         "related_information": related_information,
#         "query": user_query,
#         "dark_mode": False
#     })

##################################################################################

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
