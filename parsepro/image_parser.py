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
        self.__system_prompt = """
    
        Extract all content from the provided image or PDF and convert it to clean, structured Markdown format.

            REQUIREMENTS:
            1. Return ONLY the extracted content in proper Markdown syntax
            2. Preserve the original document structure including:
            - Headings (h1-h6)
            - Paragraphs
            - Lists (ordered and unordered)
            - Tables
            - Text formatting (bold, italic, etc.)
            - Links (maintain href attributes)
            - Image placeholders with descriptive alt text
            - Code blocks and inline code
            - Blockquotes
            - Horizontal rules

            IMPORTANT:
            - Do not include any explanatory text, comments, or meta-information about the conversion
            - Do not wrap the output in code blocks or delimiters
            - Maintain the hierarchical structure of the original document
            - Preserve header/footer information when present
            - Format tables properly with aligned columns
            - Include captions for figures and tables if present
            - For equations or mathematical notation, use proper Markdown math syntax

            This extracted content will be used for direct integration into documentation systems.
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


    def convert_image_to_markdown(self, image_path: str = None , image_url:str  = None , prompt:str = None ,) -> str:
        """
        Converts an image to Markdown format using the Together API.
        
        This function processes an image and returns its Markdown representation based on
        the provided prompt. The image can be specified either by a local file path or a URL.
        
        Args:
            prompt: Instructions for how the image should be processed or described.
            image_path (str, optional): Path to a local image file. Defaults to None.
            image_url (str, optional): URL of an image to process. Defaults to None.
            
        Returns:
            str: Markdown representation of the processed image.
            
        Note:
            Either image_path or image_url must be provided, but not both.
        """
        if not (image_path or image_url):
            raise ValueError("Either 'image_path' or 'image_url' must be provided.")
        
        if prompt is None:
            prompt = self.__system_prompt

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
                    {"type": "text", "text": prompt},
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











