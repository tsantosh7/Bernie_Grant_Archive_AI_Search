import os
import json
import io
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision

# Set up Google credentials (adjust the path as needed)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/stirunag/work/keys/velvety-citizen-453216-t7-500020178385.json"

# Base folder and content directories
base_folder = os.getcwd()
content_folder = os.path.join(base_folder, "content")

# Define folder paths
website_pdfs_folder = os.path.join(content_folder, "WEBSITE_PDFS")
website_txt_folder = os.path.join(content_folder, "WEBSITE")
wikipedia_txt_folder = os.path.join(content_folder, "WIKIPEDIA")

# PNG output folder (for PDFs only)
png_output_folder = os.path.join(base_folder, "converted_png", "WEBSITE_PDFS")
os.makedirs(png_output_folder, exist_ok=True)

# Load mappings.json (keys are filenames without the .pdf extension, values are sources)
mappings_path = os.path.join(content_folder, "mappings.json")
with open(mappings_path, "r", encoding="utf-8") as f:
    mappings = json.load(f)

# Output JSON file path.
output_json_path = os.path.join(content_folder, "website_pdf_records.json")


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
    # Return the full detected text from the first annotation, if available.
    if response.text_annotations:
        return response.text_annotations[0].description
    return ""


def process_pdf(file_path):
    """
    Processes a PDF file:
      - Converts each page to a PNG image.
      - Runs OCR on each page.
      - Saves each PNG in the designated folder.
    Returns a tuple:
      (list of extracted texts, list of relative PNG paths, total number of pages)
    """
    pages = convert_from_path(file_path, dpi=300)
    extracted_texts = []
    png_paths = []
    for i, page in enumerate(pages, start=1):
        base_name = Path(file_path).stem
        png_file_name = f"{base_name}_page_{i}.png"
        png_file_path = os.path.join(png_output_folder, png_file_name)
        page.save(png_file_path, "PNG")
        # Run OCR on the page.
        extracted_text = ocr_image_google_from_image(page)
        extracted_texts.append(extracted_text)
        # Save a relative path (e.g. "WEBSITE_PDFS/<png_file_name>")
        png_paths.append(os.path.join("WEBSITE_PDFS", png_file_name))
    return extracted_texts, png_paths, len(pages)


def process_txt(file_path):
    """
    Processes a text file:
      - Reads the text content.
    Returns a tuple:
      (list with the text content, empty png_paths list, page count as 1)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [content], [], 1


# List to store processed records.
records = []

# Iterate over each mapping key and its source
for key, source in tqdm(mappings.items(), desc="Processing files"):
    # First, try WEBSITE_PDFS (expecting a PDF)
    pdf_path = os.path.join(website_pdfs_folder, key + ".pdf")
    file_downloaded = None
    extracted_text = []
    png_paths = []
    page_number = 0

    if os.path.exists(pdf_path):
        file_downloaded = pdf_path
        try:
            extracted_text, png_paths, page_number = process_pdf(pdf_path)
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            extracted_text, png_paths, page_number = [], [], 0
    else:
        # Not found in PDFs. Check WEBSITE (expecting a text file)
        website_txt_path = os.path.join(website_txt_folder, key + ".txt")
        if os.path.exists(website_txt_path):
            file_downloaded = website_txt_path
            try:
                extracted_text, png_paths, page_number = process_txt(website_txt_path)
            except Exception as e:
                print(f"Error processing text file {website_txt_path}: {e}")
                extracted_text, png_paths, page_number = [], [], 0
        else:
            # Check WIKIPEDIA folder for a text file.
            wikipedia_txt_path = os.path.join(wikipedia_txt_folder, key + ".txt")
            if os.path.exists(wikipedia_txt_path):
                file_downloaded = wikipedia_txt_path
                try:
                    extracted_text, png_paths, page_number = process_txt(wikipedia_txt_path)
                except Exception as e:
                    print(f"Error processing text file {wikipedia_txt_path}: {e}")
                    extracted_text, png_paths, page_number = [], [], 0
            else:
                print(f"File for key '{key}' not found in any folder.")
                file_downloaded = None
                extracted_text, png_paths, page_number = [], [], 0

    record = {
        "Name": key,
        "Page number": page_number,
        "Location": None,
        "file_downloaded": file_downloaded,
        "source": source,
        "extracted_text": extracted_text,
        "png_path": png_paths
    }
    records.append(record)

    # Write the updated records list to JSON in every iteration.
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

print("Records populated and saved at:", output_json_path)
