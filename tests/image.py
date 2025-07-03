from parsepro.image_parser import ImageToMarkdown
from dotenv import load_dotenv
load_dotenv()
image = ImageToMarkdown(provider="openai")

output = image.convert_image_to_markdown(image_path="docs/images/UIScoring.jpeg")
print(output)