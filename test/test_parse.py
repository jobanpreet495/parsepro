from parsepro.parsepro import ImageToMarkdown

# Initialize the converter with an API key
converter = ImageToMarkdown(api_key="deb0b57b8ea6f67a5caf84b5944d3bdfe9fd6d6b603d9bbeccd5dbc936f39d2f")

# Convert the image and get Markdown content
markdown_content = converter.convert_image_to_markdown("/home/erginous/Desktop/Invoice/a.png")
print(markdown_content)
