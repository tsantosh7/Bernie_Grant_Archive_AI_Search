import os
import re
from tqdm import tqdm
import json
import io
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/stirunag/work/keys/velvety-citizen-453216-t7-500020178385.json"

# Base folder path
base_folder = os.getcwd()
input_base = os.path.join(base_folder, "content")
text_output_base = os.path.join(base_folder, "extracted_text")
png_output_base = os.path.join(base_folder, "converted_png")

# JSON file from the previous download step.
json_input_path = os.path.join(input_base, "downloaded_files.json")

# List of folder names to process
folders = [
    "FIGHTING RACISM",
    "POLITICS_ BLACK INITIATIVES",
    "TRADE UNIONISM",
    "INTERNATIONALISM",
    "POLICING",
    "REPARATIONS",
]


def ocr_image_google_from_image(image_obj):
    """
    Runs Google Cloud Vision OCR on a Pillow image object.
    Returns the extracted text.
    """
    client = vision.ImageAnnotatorClient()
    buffered = io.BytesIO()
    image_obj.save(buffered, format="PNG")
    content = buffered.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    # The first annotation typically contains the full detected text.
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""


def process_tif(file_path, folder_name):
    """
    Processes a TIFF/TIF file:
      - Converts it to PNG.
      - Runs OCR on it.
      - Saves the PNG and extracted text.
    Returns:
      extracted_text (str) and relative PNG path (str, e.g. "FIGHTING RACISM/filename.png")
    """
    base_name = Path(file_path).stem
    # Create output directories if they don't exist.
    text_out_dir = os.path.join(text_output_base, folder_name)
    png_out_dir = os.path.join(png_output_base, folder_name)
    os.makedirs(text_out_dir, exist_ok=True)
    os.makedirs(png_out_dir, exist_ok=True)

    # Open the TIFF image.
    image = Image.open(file_path)
    # Save converted PNG.
    png_file_name = f"{base_name}.png"
    png_file_path = os.path.join(png_out_dir, png_file_name)
    image.save(png_file_path, "PNG")

    # Extract text using Google Cloud Vision OCR.
    extracted_text = ocr_image_google_from_image(image)
    # Save extracted text.
    text_file_path = os.path.join(text_out_dir, f"{base_name}.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"Processed TIFF: {file_path}")
    # Return extracted text and relative PNG path (FolderName/filename.png)
    relative_png = os.path.join(folder_name, png_file_name)
    return extracted_text, relative_png


def process_pdf(file_path, folder_name):
    """
    Processes a PDF file:
      - Converts each page to PNG.
      - Runs OCR on each page.
      - Saves each PNG and the corresponding extracted text.
    Returns:
      extracted_texts (list of str) and png_paths (list of relative paths, e.g. "FIGHTING RACISM/filename_page_1.png")
    """
    base_name = Path(file_path).stem
    text_out_dir = os.path.join(text_output_base, folder_name)
    png_out_dir = os.path.join(png_output_base, folder_name)
    os.makedirs(text_out_dir, exist_ok=True)
    os.makedirs(png_out_dir, exist_ok=True)

    # Convert PDF pages to images (set dpi as needed).
    pages = convert_from_path(file_path, dpi=300)
    extracted_texts = []
    png_paths = []
    for i, page in enumerate(pages, start=1):
        # Save each page as PNG.
        png_file_name = f"{base_name}_page_{i}.png"
        png_file_path = os.path.join(png_out_dir, png_file_name)
        page.save(png_file_path, "PNG")
        # Run OCR on the page.
        extracted_text = ocr_image_google_from_image(page)
        extracted_texts.append(extracted_text)
        # Save extracted text.
        text_file_path = os.path.join(text_out_dir, f"{base_name}_page_{i}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        png_paths.append(os.path.join(folder_name, png_file_name))
    print(f"Processed PDF: {file_path}")
    return extracted_texts, png_paths


def process_file(file_path, folder_name):
    """
    Processes a file based on its extension.
    Returns a tuple (extracted_text, png_path) or (list, list) for PDF.
    If file type is unsupported, returns (None, None).
    """
    ext = Path(file_path).suffix.lower()
    if ext in [".tif", ".tiff"]:
        return process_tif(file_path, folder_name)
    elif ext == ".pdf":
        return process_pdf(file_path, folder_name)
    else:
        print(f"Skipping unsupported file: {file_path}")
        return None, None


# Load the JSON file with downloaded records.
with open(json_input_path, "r", encoding="utf-8") as f:
    download_records = json.load(f)

# Process each record in the JSON file.
# We assume that "file_downloaded" is an absolute path like
# "/home/stirunag/work/github/RAG/data/content/<folder>/filename.ext"
for record in tqdm(download_records):
    file_path = record.get("file_downloaded", "")
    if file_path:
        try:
            # Extract folder name from file_path (assumes folder is the second last component).
            parts = Path(file_path).parts
            folder_name = parts[-2] if len(parts) >= 2 else ""
            extracted_text, png_path = process_file(file_path, folder_name)
            record["extracted_text"] = extracted_text
            record["png_path"] = png_path
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print("No file_downloaded for record:", record.get("Name", ""))

    # Immediately update the JSON file after processing each record
    with open(json_input_path, "w", encoding="utf-8") as f:
        json.dump(download_records, f, indent=4, ensure_ascii=False)

print("Processing complete and JSON file updated with extracted text and PNG paths.")
