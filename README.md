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
- Together API key (required for authentication)/ Openai Key/ Groq Key


## Installation

```bash
!apt-get install poppler-utils
pip install parsepro
```

## Usage for Image
```bash
from parsepro import ImageToMarkdown

# Initialize the client with your preferred provider and API key.
# Supported providers: "together", "openai", "groq"
# You can either pass the API key directly or set it via environment variable:
# - TOGETHER_API_KEY for Together
# - OPENAI_API_KEY for OpenAI
# - GROQ_API_KEY for Groq

import os
# Example: os.environ['TOGETHER_API_KEY'] = "your_api_key_here"

image_to_markdown = ImageToMarkdown(provider="together")  # or "openai", "groq"

# Convert an image to Markdown.
# You can pass either a local file path or a remote image URL.
# You can also override the default system prompt if needed.

markdown_content = image_to_markdown.convert_image_to_markdown(
    image_path="path/to/your/image.jpg"  # or image_url="https://example.com/image.png"
    # prompt="Custom prompt here..."  # optional
)

print(markdown_content)

```


## Usage for pdf
```bash
from parsepro import PDFToMarkdown

# Initialize the client with your Together API key
# Note: You can also set your API key as an environment variable named TOGETHER_API_KEY.
# import os 
# os.environ['TOGETHER_API_KEY'] = ""

pdf_to_markdown = PDFToMarkdown()

# Convert an image to Markdown
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(pdf_path = "path/to/your/your_pdf.pdf") # pdf_url = "" and pages_to_parse = "2" or range "2-8"
print(markdown_content)
```


## Define  custom prompt

```bash

# Specify prompt for your usecase

# Convert an image to Markdown
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(pdf_path = "path/to/your/your_pdf.pdf", prompt = "") # pdf_url = "" and pages_to_parse = "2" or range "2-8"

markdown_content = pdf_to_markdown.convert_pdf_to_markdown(pdf_path = "path/to/your/your_pdf.pdf",prompt = "")

```
