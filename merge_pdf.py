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
        "ch1/README.pdf",
        "ch2/README.pdf",
        "ch3/README.pdf",
        "ch4/README.pdf",
        "ch4_1/README.pdf",
        # "ch5/README.pdf",
        "ch6/README.pdf",
        "ch7/README.pdf",
        "ch8/README.pdf",
        "ch9/README.pdf",
        "ch10/README.pdf",
        "ch11/README.pdf",
        "ch12/README.pdf",
        "ch99/gym_intro_1.pdf",
        "ch99/a2c.pdf",
        "ch99/dqn.pdf",
        "ch99/ddpg.pdf",
    ],
}


if __name__ == "__main__":
    item_name = BOOK_INFO['item']
    root_dir = f"docs/{item_name}"
    output_pdf = f"{root_dir}/{item_name}.pdf"
    chapter_pdfs = [f"{root_dir}/{chapter_pdf}" for chapter_pdf in BOOK_INFO['chapters'] if chapter_pdf.endswith('.pdf')]
    if chapter_pdfs:
        merge_pdfs(chapter_pdfs, output_pdf)
        print(f"Merged {len(chapter_pdfs)} PDFs into {output_pdf}")
    else:
        print("No PDF files found.")