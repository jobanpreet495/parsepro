# ParsePro

ParsePro is a Python library that converts images into Markdown format using the Together API. It leverages large language models (Llama-3.2) to extract content from images and structure it in a readable Markdown format .

## Features

- Converts image content into Markdown format.
- Preserves structure, including headers, footers, images, tables, and subtext.
- Easy to use with a single function call.



## Requirements

- Python 3.10+
- Together API key (required for authentication)


## Installation

Install the package with:

```bash
pip install parsepro
```


## Usage
```bash
from parsepro import ImageToMarkdown

# Initialize the client with your Together API key
# Note: You can also set your API key as an environment variable named TOGETHER_API_KEY.
# import os 
# os.environ['TOGETHER_API_KEY'] = ""

image_to_markdown = ImageToMarkdown(api_key="your_api_key_here")

# Convert an image to Markdown
markdown_content = image_to_markdown.convert_image_to_markdown("path/to/your/image.jpg")
print(markdown_content)
```

