from parsepro.parsepro import ImageToMarkdown

# Initialize the converter with an API key
converter = ImageToMarkdown(api_key="your togetherai api key")

# Convert the image and get Markdown content
markdown_content = converter.convert_image_to_markdown("/home/erginous/Desktop/Invoice/a.png")
print(markdown_content)
