from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

pdf_folder = Path("pdf_chunks")

for pdf_file in pdf_folder.glob("*.pdf"):

    print(f"Traitement : {pdf_file.name}")

    result = converter.convert(str(pdf_file))

    markdown = result.document.export_to_markdown()

    output_md = pdf_file.with_suffix(".md")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Créé : {output_md.name}")