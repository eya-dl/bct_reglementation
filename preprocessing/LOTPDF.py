from PyPDF2 import PdfReader, PdfWriter
import os

# PDF source
input_pdf = "reglementation.pdf"

# Dossier de sortie
output_folder = "pdf_chunks"
os.makedirs(output_folder, exist_ok=True)

# Lecture du PDF
reader = PdfReader(input_pdf)
total_pages = len(reader.pages)

# Taille d'un lot
chunk_size = 40

print(f"Nombre total de pages : {total_pages}")

# Découpage
for start in range(0, total_pages, chunk_size):

    end = min(start + chunk_size, total_pages)

    writer = PdfWriter()

    for page_num in range(start, end):
        writer.add_page(reader.pages[page_num])

    output_file = os.path.join(
        output_folder,
        f"part_{start+1}_{end}.pdf"
    )

    with open(output_file, "wb") as f:
        writer.write(f)

    print(f"Créé : {output_file}")

print("Découpage terminé !")