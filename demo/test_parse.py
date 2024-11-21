from parsepro.image_parser import ImageToMarkdown
from parsepro.pdf_parser import PDFToMarkdown
from dotenv import load_dotenv 
load_dotenv()
# Initialize the converter with an API key
converter = PDFToMarkdown()
markdown = converter.convert_pdf_to_markdown(pdf_path = "documents/invoice.pdf")
# Convert the image and get Markdown content




print(markdown)
