from parsepro.pdf_parser import PDFToMarkdown
from dotenv import load_dotenv
load_dotenv()
pdf = PDFToMarkdown(provider="openai")

output = pdf.convert_pdf_to_markdown(pdf_path="/home/erginous/Desktop/VADI_RESEARCH/parsepro/docs/facture f.pdf")
print(output)