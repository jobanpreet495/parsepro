# ParsePro

**ParsePro** is a powerful Python library that transforms images and PDFs into clean, structured Markdown using state-of-the-art vision-language models. Whether you're working with screenshots, scanned documents, or reference materials, ParsePro preserves the original content structure while converting it to Markdown format.

## 🔍 Overview

ParsePro seamlessly integrates with multiple AI providers—**Together**, **OpenAI**, **Groq**, and now **Anthropic**—leveraging cutting-edge models like:

- **Llama-3.2-11B-Vision**
- **GPT-4o**
- **Llama-4-scout-17b-16e-instruct**
- **claude-3-7-sonnet**

The library accurately preserves document elements including:

- ✓ Headings and paragraph structures
- ✓ Ordered and unordered lists
- ✓ Tables with proper formatting
- ✓ Code blocks with syntax highlighting
- ✓ Inline code segments
- ✓ Links and references
- ✓ Images with appropriate placeholders
- ✓ Blockquotes
- ✓ Mathematical notation

Perfect for researchers, developers, content creators, and anyone looking to integrate scanned or image-based content into modern Markdown-based workflows.

## ✨ Features

- **Image Processing**
  - Local image parsing
  - Remote image URL support
  - Structure preservation

- **PDF Handling**
  - Single-page and multi-page document support
  - Local PDF file processing
  - Remote PDF URL support
  - Page range specification
  
- **Provider Flexibility**
  - Multiple AI provider support (Together, OpenAI, Groq, Anthropic)
  - Customizable model parameters
  - JSON output option

- **Customization**
  - Custom prompt support
  - Configurable parameters (temperature, etc.)

## 📋 Requirements

- **Python 3.10+**
- **poppler-utils** (required for PDF processing)
- One of the following API keys:
  - Together API key
  - OpenAI API key
  - Groq API key
  - Anthropic API key

## 🔧 Installation

### Step 1: Install poppler-utils

#### Ubuntu/Debian
```bash
sudo apt-get install -y poppler-utils
```

#### macOS
```bash
brew install poppler
```

#### Windows
Download the latest poppler release from [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add it to your PATH.

### Step 2: Install ParsePro

```bash
pip install parsepro
```

## 🚀 Usage

### Image to Markdown Conversion

```python
from parsepro import ImageToMarkdown
import os

# Set API key via environment variable
os.environ['TOGETHER_API_KEY'] = "your_api_key_here"
# Alternatively: os.environ['OPENAI_API_KEY'] = "your_openai_key"
# Alternatively: os.environ['GROQ_API_KEY'] = "your_groq_key"
# Alternatively: os.environ['ANTHROPIC_API_KEY'] = "your_anthropic_key"

# Initialize with preferred provider
image_to_markdown = ImageToMarkdown(provider="together")  # Options: "together", "openai", "groq", "anthropic"

# Convert from local file
markdown_content = image_to_markdown.convert_image_to_markdown(
    image_path="path/to/your/image.jpg",
    kwargs={"json": False, "temperature": 0.3}  # Optional parameters
)

# Convert from URL
markdown_content = image_to_markdown.convert_image_to_markdown(
    image_url="https://example.com/image.png",
    prompt="Custom instruction for processing this image",  # Optional
    kwargs={"json": True, "temperature": 0.7}  # Optional parameters
)

print(markdown_content)
```

### PDF to Markdown Conversion

```python
from parsepro import PDFToMarkdown
import os

# Set API key via environment variable
os.environ['TOGETHER_API_KEY'] = "your_api_key_here"

# Initialize the PDF converter
pdf_to_markdown = PDFToMarkdown(provider="together")  # Default is "together"

# Convert entire PDF from local file
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(
    pdf_path="path/to/your/document.pdf"
)

# Convert specific pages from remote PDF
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(
    pdf_url="https://example.com/document.pdf",
    pages_to_parse="2-5",  # Process pages 2 through 5
    prompt="Extract technical specifications and code examples",  # Optional custom prompt
    kwargs={"temperature": 0.2}  # Optional parameters
)

print(markdown_content)
```

### Using Custom Prompts

You can provide custom instructions to guide the AI model's extraction process:

```python
# Example with custom prompt for technical document extraction
markdown_content = image_to_markdown.convert_image_to_markdown(
    image_path="path/to/technical_diagram.png",
    prompt="Extract all technical specifications and represent any diagrams as markdown tables. Include all numerical values and units."
)

# Example for parsing a scientific paper
markdown_content = pdf_to_markdown.convert_pdf_to_markdown(
    pdf_path="path/to/scientific_paper.pdf",
    prompt="Focus on methodology and results sections. Format mathematical equations using LaTeX notation. Create proper tables for all results data.",
    kwargs={"json": True, "temperature": 0.2}
)
```

## 🛠️ Advanced Configuration

### Model Parameters

The `kwargs` parameter allows passing additional parameters to the underlying AI model:

```python
# Example with extended configuration
markdown_content = image_to_markdown.convert_image_to_markdown(
    image_path="path/to/your/image.jpg",
    kwargs={
        "json": True,        # Return JSON output
        "temperature": 0.7,  # Control randomness (0.0 to 1.0)
        # Other provider-specific parameters can be added here
    }
)
```



Made with ❤️ by the ParsePro team
