# Bernie Grant Digital Archive – Generative AI Search

This repository contains a FastAPI application that demonstrates retrieval-augmented generation (RAG) for the Bernie Grant digital archive. The system uses a [FAISS](https://github.com/facebookresearch/faiss) index to retrieve relevant documents, then calls OpenAI’s GPT models to produce an AI-generated summary or chat answer referencing those documents.

## Features

- **AI-Powered Retrieval**: Quickly searches through archival documents for relevant passages.
- **FastAPI Web Interface**: Provides a simple front-end with a search box and optional example queries.
- **Reference Linking**: Display inline references ([1], [2]) for sources, each linked to the original document URL.
- **Expandable Snippets**: Shows short snippets from each document, with optional images rendered in a lightbox.

## Requirements

- **Python 3.8+**
- A local FAISS index file (`combined_index.faiss`) and a matching metadata JSON (`combined_metadata.json`)
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

## Installation

1. **Clone** the repository:
```bash
git clone https://github.com/tsantosh7/Bernie_Grant_Archive_AI_Search
cd <Bernie_Grant_Archive_AI_Search>
cd <Chatbot>
```

2. **Create and activate** a virtual environment (recommended):
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install** dependencies:
```bash
pip install -r requirements.txt
```

4. **Set** your OpenAI API Key. Two options:
   - Create a `.env` file in the root directory with:
     ```
     OPENAI_API_KEY=your-openai-api-key-here
     ```
   - Or export it as an environment variable (e.g. `export OPENAI_API_KEY=...` on Linux/macOS).

5. **Place** your FAISS index and metadata files in the correct folder:
   - By default, the code expects `combined_index.faiss` and `combined_metadata.json` in `../models/openAI/` relative to your main code folder.

## Usage

1. **Run** the FastAPI application (with `uvicorn`):
```bash
uvicorn main:app --reload
```
   - Replace `main:app` if your main Python file or FastAPI app instance has a different name.

2. **Open** your browser at: http://127.0.0.1:8000

   - You’ll see a page with a search box and example queries.

3. **Enter** a query in the search box:
   - Click “AI Generated Response” to call GPT in a generation mode with context from retrieved documents.
   - The system returns an answer plus a “Source & Related Information” section listing top retrieved documents and Sources.

## Application Structure

- **chat_app.py** – FastAPI entry point containing:
  - Embedding & retrieval logic
  - Routes for `/` and `/query_form`
  - OpenAI prompt construction
- **templates/index.html** – Jinja2 template for the web UI
- **static/** – Contains CSS, images, or other static assets (e.g. logos)
- **models/openAI/** – Stores:
  - `combined_index.faiss` – The FAISS index
  - `combined_metadata.json` – Document metadata (source URLs, snippet text, etc.)

## Customization

- **Styling**: Modify `templates/index.html` and any CSS to fit your design.
- **Prompt**: Adjust the prompt text in the `query_form` route for custom GPT responses.
- **Reference Linking**: If you use a function like `make_inline_references_clickable`, you can customize how `[1]`, `[2]` get turned into hyperlinks.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for bug fixes, new features, or improvements.

## License

Some rights reserved.
---

**Enjoy exploring the Bernie Grant Archive with AI-assisted search!**

