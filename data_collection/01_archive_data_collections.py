import pandas as pd
import os
import re
import json


# Base folder path
base_folder = os.getcwd()
csv_folder = os.path.join(base_folder, "archive")
output_base = os.path.join(base_folder, "content")

# List of CSV files
csv_files = [
    "Bernie Grant digital archive inventory - INTERNATIONALISM.csv",
    "Bernie Grant digital archive inventory - POLICING.csv",
    "Bernie Grant digital archive inventory - REPARATIONS.csv",
    "Bernie Grant digital archive inventory - FIGHTING RACISM.csv",
    "Bernie Grant digital archive inventory - POLITICS_ BLACK INITIATIVES.csv",
    "Bernie Grant digital archive inventory - TRADE UNIONISM.csv"
]

# Ensure base output directory exists
os.makedirs(output_base, exist_ok=True)


def safe_filename(s):
    """
    Sanitizes a string to be safe for filenames.
    Replaces any character that is not alphanumeric, dash, or underscore with an underscore.
    """
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)


def extract_source(name):
    """
    Extracts the source from the name.
    The source is assumed to be the content of the last square brackets in the name,
    with all non-alphanumeric characters removed.
    """
    # Find all content inside square brackets
    bracket_contents = re.findall(r'\[([^\]]+)\]', name)
    if bracket_contents:
        # Use the last occurrence
        raw_source = bracket_contents[-1]
        # Remove any non-alphanumeric characters
        source = re.sub(r'[^a-zA-Z0-9]', '', raw_source)
        return source
    return ""


# Function to download file using wget
def download_file_wget(file_id, output_file):
    """Downloads a Google Drive file using wget"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        os.system(f"wget --no-check-certificate '{url}' -O '{output_file}'")
    except Exception as e:
        print(f"Error downloading {output_file}: {e}")


# List to store information about downloaded files
download_records = []

# Process each CSV file
for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)

    # Extract folder name from CSV filename
    folder_name = csv_file.replace("Bernie Grant digital archive inventory - ", "").replace(".csv", "").strip()
    output_folder = os.path.join(output_base, folder_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file and fill NaN with empty strings
    df = pd.read_csv(csv_path, delimiter=",").fillna("")

    # Process each row assuming columns: Name (0), Page number (1), Location (2)
    for index, row in df.iterrows():
        # Use iloc to access by position and sanitize the title
        name = str(row.iloc[0]).strip()
        title = safe_filename(name)
        drive_link = str(row.iloc[2]).strip()
        page_number = row.iloc[1]

        # Skip rows with empty title or invalid link
        if not title or not drive_link or "drive.google.com" not in drive_link:
            print(f"Skipping invalid entry: {title}")
            continue

        try:
            # Extract Google Drive file ID
            file_id = drive_link.split('/d/')[1].split('/view')[0]

            # Choose file extension based on page number (using iloc)
            if int(page_number) > 1:
                output_file = os.path.join(output_folder, f"{title}.pdf")
            else:
                output_file = os.path.join(output_folder, f"{title}.tif")

            print(f"Downloading: {title}...")
            download_file_wget(file_id, output_file)

            # Derive the source from the original name
            source = extract_source(name)

            # Add record to our list
            record = {
                "Name": name,
                "Page number": page_number,
                "Location": drive_link,
                "file_downloaded": output_file,
                "source": source
            }
            download_records.append(record)

        except Exception as e:
            print(f"Error processing {title}: {e}")

print("Download complete.")

# Write the records to a JSON file in the output base folder
json_output_path = os.path.join(output_base, "downloaded_files.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(download_records, f, indent=4, ensure_ascii=False)

print(f"Download records saved to {json_output_path}.")
