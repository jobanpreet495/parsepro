import os
import base64
import logging
from together import Together
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ImageToMarkdown:
    """
    A class to convert images into Markdown format.
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the ImageToMarkdown client.
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable or pass as an argument.")
        
        self.client = Together(api_key=self.api_key)
        self.system_prompt = """
    
        Convert the provided image into Markdown format. Ensure that all content from the page is included, such as headers, footers, subtexts, images (with alt text if possible), tables, and any other elements.

        Requirements:
        - Output Only Markdown: Return solely the Markdown content without any additional explanations or comments.
        - No Delimiters: Do not use code fences or delimiters like ```markdown.
        - Complete Content: Do not omit any part of the page, including headers, footers, and subtext.

        Note :  Do not output additional  explanations or comments.  
        """
        logger.info("ImageToMarkdown initialized with Together API client.")

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Encodes an image to a base64 string.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                logger.info(f"Encoding image: {image_path}")
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            logger.error(f"Error reading image file: {image_path}")
            raise

    @staticmethod
    def _download_image_to_base64(image_url: str) -> str:
        """
        Downloads an image from a URL and encodes it to a base64 string.
        """
        try:
            logger.info(f"Downloading image from URL: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except requests.RequestException as e:
            logger.error(f"Failed to download image from URL: {image_url}", exc_info=True)
            raise ValueError("Failed to download image from URL.") from e


    def convert_image_to_markdown(self, image_path: str = None , image_url:str  = None) -> str:
        """
        Converts an image to Markdown format using the Together API.
        
        The user can provide either a local `image_path` or an `image_url`.
        """
        if not (image_path or image_url):
            raise ValueError("Either 'image_path' or 'image_url' must be provided.")

        if image_path:
            logger.info("Processing local image path.")
            base64_image = self._encode_image(image_path)
        elif image_url:
            logger.info("Processing image URL.")
            base64_image = self._download_image_to_base64(image_url)
        
        # Prepare the API message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]
        
        try:
            logger.info("Sending request to Together API for Markdown conversion.")
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                temperature = 0
            )
            logger.info("Received response from Together API.")
        except Exception as e:
            logger.error("Failed to get a response from the Together API", exc_info=True)
            raise

        try:
            return response.choices[0].message.content
        except (IndexError, AttributeError) as e:
            logger.error("Invalid response format from the Together API", exc_info=True)
            raise ValueError("Invalid response format from the Together API") from e











