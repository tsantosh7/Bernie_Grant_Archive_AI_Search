# import json
# import faiss
# import numpy as np
# from tqdm import tqdm
# import openai
# from pathlib import Path
# import os
# import hashlib
# from dotenv import load_dotenv
#
# # Uncomment if you are using FastAPI (here not strictly needed for indexing)
# # from fastapi import FastAPI
#
# load_dotenv()  # Loads variables from .env into the environment
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# if openai.api_key is None:
#     print("API key not found. Please set the OPENAI_API_KEY environment variable.")
# else:
#     print("API key found.")
#
# base_folder = os.path.dirname(os.getcwd())
# data_folder = os.path.join(base_folder, "data_collection/content/")
# # Paths to the two JSON files (assumed to be JSON arrays)
# json_files = [
#     os.path.join(data_folder, "downloaded_files.json"),
#     os.path.join(data_folder, "website_pdf_records.json")
# ]
#
# # Output directory for FAISS indexes and metadata
# output_dir = os.path.join(base_folder, "models", "openAI")
# os.makedirs(output_dir, exist_ok=True)
#
# def generate_unique_id(text):
#     """Generate a unique ID based on the text (or any identifier)."""
#     return hashlib.sha256(text.encode()).hexdigest()
#
# def clean_name(name):
#     """
#     For downloaded_files.json records: if the name ends with a square-bracketed phrase,
#     remove it. e.g., "Bernie Grant [text]" -> "Bernie Grant"
#     """
#     if "[" in name and name.endswith("]"):
#         idx = name.rfind("[")
#         return name[:idx].strip()
#     return name
#
# # Prepare lists for page-level and document-level indexing
# page_texts = []
# page_metadata = []
#
# doc_texts = []
# doc_metadata = []
#
# # Process each JSON file
# for file_path in json_files:
#     print(f"Processing file: {file_path}")
#     with open(file_path, "r", encoding="utf-8") as f:
#         records = json.load(f)  # assume file is a JSON array of records
#
#     for record in records:
#         # Update the source and add search_source
#         name = record.get("Name", "")
#         if file_path.endswith("downloaded_files.json"):
#             # Prepend the URL prefix to source
#             source = record.get("source", "")
#             record["source"] = "https://berniegrantarchive.org.uk/" + source
#             cleaned_name = clean_name(name)
#             record["search_source"] = "https://berniegrantarchive.org.uk/?s=" + cleaned_name
#         else:
#             record["search_source"] = "https://berniegrantarchive.org.uk/?s=" + name
#
#         # Process extracted text: if already a list, use it; if string, wrap it into a list.
#         extracted = record.get("extracted_text", "")
#         if isinstance(extracted, list):
#             pages_list = extracted
#         elif isinstance(extracted, str):
#             pages_list = [extracted]
#         else:
#             pages_list = []
#
#         # Create whole-document text by concatenating pages.
#         whole_doc_text = "\n\n".join(pages_list)
#         # Generate a document-level unique id (based on Name and source)
#         doc_id = record.get("_id", generate_unique_id(name + record.get("source", "")))
#         doc_metadata.append({
#             "_id": doc_id,
#             "Name": name,
#             "source": record.get("source", ""),
#             "search_source": record.get("search_source", ""),
#             "extracted_text": whole_doc_text,
#             "png_info": record.get("png_path", []),
#             "level": "document"
#         })
#         doc_texts.append(whole_doc_text)
#
#         # For page-level indexing, add each page separately.
#         for idx, page in enumerate(pages_list):
#             page_id = generate_unique_id(name + str(idx))
#             # You can choose to combine the name and the page text.
#             combined_text = f"{name}\n\n{page}"
#             page_texts.append(combined_text)
#             page_metadata.append({
#                 "_id": page_id,
#                 "Name": name,
#                 "source": record.get("source", ""),
#                 "search_source": record.get("search_source", ""),
#                 "extracted_text": page,
#                 "png_info": record.get("png_path", []),
#                 "page_number": idx + 1,
#                 "level": "page"
#             })
#
# print(f"Total page entries: {len(page_texts)}")
# print(f"Total document entries: {len(doc_texts)}")
#
# # # Function to get embedding from OpenAI for a given text
# def get_embedding(text, model="text-embedding-ada-002"):
#     # response = openai.Embedding.create(
#     #     input=text,
#     #     model=model
#     # )
#     # return response["data"][0]["embedding"]
#     return np.random.rand(1536).tolist()
#
#
#
#
#
# # Compute embeddings for page-level texts
# page_embeddings = []
# for text in tqdm(page_texts, desc="Computing page embeddings"):
#     try:
#         emb = get_embedding(text)
#         page_embeddings.append(emb)
#     except Exception as e:
#         print("Error computing embedding for a page:", e)
#         # Append a zero vector (dimension 1536 for ada-002) if there is an error.
#         page_embeddings.append([0] * 1536)
# page_embeddings = np.array(page_embeddings).astype("float32")
# embedding_dim = page_embeddings.shape[1]
#
# # Build the FAISS index for page-level embeddings using L2 distance.
# page_index = faiss.IndexFlatL2(embedding_dim)
# page_index.add(page_embeddings)
#
# page_index_file = os.path.join(output_dir, "index_page.faiss")
# page_metadata_file = os.path.join(output_dir, "metadata_page.json")
# faiss.write_index(page_index, page_index_file)
# with open(page_metadata_file, "w", encoding="utf-8") as f:
#     json.dump(page_metadata, f, indent=2)
#
# print("✅ Page-level indexing complete!")
#
# # Compute embeddings for whole-document texts
# doc_embeddings = []
# for text in tqdm(doc_texts, desc="Computing document embeddings"):
#     try:
#         emb = get_embedding(text)
#         doc_embeddings.append(emb)
#     except Exception as e:
#         print("Error computing embedding for a document:", e)
#         doc_embeddings.append([0] * 1536)
# doc_embeddings = np.array(doc_embeddings).astype("float32")
# embedding_dim_doc = doc_embeddings.shape[1]
#
# # Build the FAISS index for document-level embeddings
# doc_index = faiss.IndexFlatL2(embedding_dim_doc)
# doc_index.add(doc_embeddings)
#
# doc_index_file = os.path.join(output_dir, "index_document.faiss")
# doc_metadata_file = os.path.join(output_dir, "metadata_document.json")
# faiss.write_index(doc_index, doc_index_file)
# with open(doc_metadata_file, "w", encoding="utf-8") as f:
#     json.dump(doc_metadata, f, indent=2)
#
# print("✅ Document-level indexing complete!")
#
#
#
#
import json
import faiss
import numpy as np
from tqdm import tqdm
import openai
from pathlib import Path
import os
import hashlib
from dotenv import load_dotenv

# Uncomment if you are using FastAPI (here not strictly needed for indexing)
# from fastapi import FastAPI

load_dotenv()  # Loads variables from .env into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    print("API key found.")

# Set base_folder to the parent folder of the current working directory using os.path.dirname
base_folder = os.path.dirname(os.getcwd())
data_folder = os.path.join(base_folder, "data_collection/content/")
# Paths to the two JSON files (assumed to be JSON arrays)
json_files = [
    os.path.join(data_folder, "downloaded_files.json"),
    os.path.join(data_folder, "website_pdf_records.json")
]

# Output directory for FAISS index and metadata
output_dir = os.path.join(base_folder, "models", "openAI")
os.makedirs(output_dir, exist_ok=True)

def generate_unique_id(text):
    """Generate a unique ID based on the text (or any identifier)."""
    return hashlib.sha256(text.encode()).hexdigest()

def clean_name(name):
    """
    For downloaded_files.json records: if the name ends with a square-bracketed phrase,
    remove it. e.g., "Bernie Grant [text]" -> "Bernie Grant"
    """
    if "[" in name and name.endswith("]"):
        idx = name.rfind("[")
        return name[:idx].strip()
    return name

# Prepare lists for page-level and document-level indexing
page_texts = []
page_metadata = []

doc_texts = []
doc_metadata = []

# Process each JSON file
for file_path in json_files:
    print(f"Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)  # assume file is a JSON array of records

    for record in records:
        name = record.get("Name", "")
        # Update source and search_source based on PNG info.
        if file_path.endswith("downloaded_files.json"):
            # Prepend the URL prefix to source
            source = record.get("source", "")
            record["source"] = "https://berniegrantarchive.org.uk/" + source
            png_info = record.get("png_path")
            # Check if png_info is None or empty
            if png_info is None or (isinstance(png_info, str) and png_info.strip() == ""):
                record["search_source"] = ""
            else:
                cleaned_name = clean_name(name)
                record["search_source"] = "https://berniegrantarchive.org.uk/?s=" + cleaned_name
        else:
            # For website_pdf_records.json, we assume the PNG info is in "png_path"
            png_info = record.get("png_path")
            if png_info is None:
                record["search_source"] = ""
            elif isinstance(png_info, list):
                if not png_info:
                    record["search_source"] = ""
                else:
                    record["search_source"] = "https://berniegrantarchive.org.uk/?s=" + name
            elif isinstance(png_info, str):
                if png_info.strip() == "":
                    record["search_source"] = ""
                else:
                    record["search_source"] = "https://berniegrantarchive.org.uk/?s=" + name
            else:
                record["search_source"] = ""

        # Process extracted text: if already a list, use it; if string, wrap it into a list.
        extracted = record.get("extracted_text", "")
        if isinstance(extracted, list):
            pages_list = extracted
        elif isinstance(extracted, str):
            pages_list = [extracted]
        else:
            pages_list = []

        # Create whole-document text by concatenating pages.
        whole_doc_text = "\n\n".join(pages_list)
        # Generate a document-level unique id (based on Name and source)
        doc_id = record.get("_id", generate_unique_id(name + record.get("source", "")))
        doc_metadata.append({
            "_id": doc_id,
            "Name": name,
            "source": record.get("source", ""),
            "search_source": record.get("search_source", ""),
            "extracted_text": whole_doc_text,
            "png_info": record.get("png_path", []),
            "level": "document"
        })
        doc_texts.append(whole_doc_text)

        # For page-level indexing, add each page separately.
        for idx, page in enumerate(pages_list):
            page_id = generate_unique_id(name + str(idx))
            # Combine the name and the page text.
            combined_text = f"{name}\n\n{page}"
            page_texts.append(combined_text)
            page_metadata.append({
                "_id": page_id,
                "Name": name,
                "source": record.get("source", ""),
                "search_source": record.get("search_source", ""),
                "extracted_text": page,
                "png_info": record.get("png_path", []),
                "page_number": idx + 1,
                "level": "page"
            })

print(f"Total page entries: {len(page_texts)}")
print(f"Total document entries: {len(doc_texts)}")

# ---- Combined Index Logic ----
# Combine the texts and metadata from page-level and document-level entries.
combined_texts = page_texts + doc_texts
combined_metadata = page_metadata + doc_metadata

# # Function to get embedding from OpenAI for a given text
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response["data"][0]["embedding"]
    # return np.random.rand(1536).tolist()

# Compute embeddings for combined texts
combined_embeddings = []
for text in tqdm(combined_texts, desc="Computing combined embeddings"):
    try:
        emb = get_embedding(text)
        combined_embeddings.append(emb)
    except Exception as e:
        print("Error computing embedding:", e)
        combined_embeddings.append([0] * 1536)
combined_embeddings = np.array(combined_embeddings).astype("float32")
embedding_dim = combined_embeddings.shape[1]

# Build a single FAISS index using L2 distance.
combined_index = faiss.IndexFlatL2(embedding_dim)
combined_index.add(combined_embeddings)

# Save the combined index and metadata to disk.
combined_index_file = os.path.join(output_dir, "combined_index.faiss")
combined_metadata_file = os.path.join(output_dir, "combined_metadata.json")
faiss.write_index(combined_index, combined_index_file)
with open(combined_metadata_file, "w", encoding="utf-8") as f:
    json.dump(combined_metadata, f, indent=2)

print("✅ Combined indexing complete!")
