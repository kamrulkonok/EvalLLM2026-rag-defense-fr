import os, zipfile, json
import fitz 
from collections import defaultdict

ZIP_PATH = "Corpus_raw.zip"
OUT_DIR = "."
OUT_FILE = "data/pages.jsonl"

os.makedirs("data", exist_ok=True)

if os.path.isfile(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(OUT_DIR)

elif os.path.isdir(ZIP_PATH):
    for f in os.listdir(ZIP_PATH):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(ZIP_PATH, f)) as z:
                z.extractall(OUT_DIR)

print("Unzip done\n")

pdf_files = []
for root, _, files in os.walk(OUT_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, f))

print(f"Total PDFs found: {len(pdf_files)}")
# process each PDF and extract text from pages
processed_pdfs = set()
failed_pdfs = 0
total_pages = 0

doc_page_count = defaultdict(int)
filename_counts = defaultdict(int)

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for path in pdf_files:
        filename = os.path.basename(path)
        filename_counts[filename] += 1

        try:
            doc = fitz.open(path)
        except Exception as e:
            failed_pdfs += 1
            continue

        processed_pdfs.add(filename)

        for i, page in enumerate(doc):
            try:
                text = page.get_text()
            except:
                continue

            if not text or not text.strip():
                continue

            record = {
                "doc_name": filename,   
                "page": i + 1,
                "text": text.strip()
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_pages += 1
            doc_page_count[filename] += 1


print(f"PDFs found        : {len(pdf_files)}")
print(f"PDFs processed    : {len(processed_pdfs)}")
print(f"PDFs failed       : {failed_pdfs}")
print(f"Total pages saved : {total_pages}")

if processed_pdfs:
    sample = next(iter(processed_pdfs))
    print(f"{sample} → {doc_page_count[sample]} pages")

print("\nDone →", OUT_FILE)