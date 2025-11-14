import os
from PyPDF2 import PdfMerger
from glob import glob

def find_pdfs(root):
    pdfs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.pdf'):
                pdfs.append(os.path.join(dirpath, f))
    return sorted(pdfs)

def merge_pdfs(pdf_list, output):
    merger = PdfMerger()
    for pdf in pdf_list:
        merger.append(pdf)
    merger.write(output)
    merger.close()

# Configuration for RL Basic book
BOOK_INFO = {
    'item': "rl_basic",
    'chapters': [
        "ch1",
        'ch2',
    ],
}


if __name__ == "__main__":
    item_name = BOOK_INFO['item']
    root_dir = f"docs/{item_name}"
    output_pdf = f"{root_dir}/{item_name}.pdf"
    chapter_pdfs = []
    for chapter in BOOK_INFO['chapters']:
        chapter_dir = f"{root_dir}/{chapter}"
        chapter_pdf = glob(f"{chapter_dir}/*.pdf")
        if chapter_pdf:
            chapter_pdfs.extend(chapter_pdf)
    if chapter_pdfs:
        merge_pdfs(chapter_pdfs, output_pdf)
        print(f"Merged {len(chapter_pdfs)} PDFs into {output_pdf}")
    else:
        print("No PDF files found.")