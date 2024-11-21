# ParsePro

ParsePro is a Python library that converts image/pdf into Markdown format using the Together API. It leverages large language models (Llama-3.2-11B-Vision) to extract content from image/pdf and structure it in a readable Markdown format .

## Features

- Local image support.
- Remote image support.
- Single-page and multi-page PDF parsing.
- Local and remote PDF file parsing.
- Page-specific parsing, where users can specify or define a page range to parse.



## Requirements

- Python 3.10+
- Together API key (required for authentication)


## For Developers

1. Clone repository

2 . Install the package with:

```bash
pip install -e .
```


## Usage for Image
```bash
from parsepro.image_parser import ImageToMarkdown

# Initialize the client with your Together API key
# Note: You can also set your API key as an environment variable named TOGETHER_API_KEY.
# import os 
# os.environ['TOGETHER_API_KEY'] = ""

image_to_markdown = ImageToMarkdown(api_key="your_api_key_here")

# Convert an image to Markdown
markdown_content = image_to_markdown.convert_image_to_markdown(image_path = "path/to/your/image.jpg")  # image_url = ""
print(markdown_content)
```


## Usage for pdf
```bash
from parsepro.pdf_parser import PDFToMarkdown

# Initialize the client with your Together API key
# Note: You can also set your API key as an environment variable named TOGETHER_API_KEY.
# import os 
# os.environ['TOGETHER_API_KEY'] = ""

pdf_to_markdown = PDFToMarkdown(api_key="your_api_key_here")

# Convert an image to Markdown
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(pdf_path = "path/to/your/your_pdf.pdf") # pdf_url = "" and pages_to_parse = "2" or range "2-8"
print(markdown_content)
```



