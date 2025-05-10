import os
import logging
import base64
from together import Together
from pdf2image import convert_from_path
from typing import List
import tempfile
import requests
from parsepro.image_parser import  ImageToMarkdown
import shutil
from urllib.parse import urlparse


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFToMarkdown:
    """
    A class to convert PDF files into Markdown format .
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the PDFToMarkdown client.
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable or pass as an argument.")

        self.client = ImageToMarkdown(api_key=self.api_key)
        logger.info("PDFToMarkdown initialized with Together API client.")

    @staticmethod
    def _is_url(path: str) -> bool:
        """
        Checks if the given path is a URL.
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def _download_pdf_from_url(pdf_url: str, temp_dir: str) -> str:
        """
        Downloads a PDF from a URL and saves it to a temporary directory.
        """
        try:
            logger.info(f"Downloading PDF from URL: {pdf_url}")
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            temp_pdf_path = os.path.join(temp_dir, "downloaded.pdf")
            with open(temp_pdf_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)
            
            logger.info(f"PDF downloaded and saved to: {temp_pdf_path}")
            return temp_pdf_path
        except Exception as e:
            logger.error(f"Failed to download PDF from URL: {pdf_url}", exc_info=True)
            raise

    @staticmethod
    def _parse_page_range(page_range: str, total_pages: int) -> List[int]:
        """
        Parses the page range string (e.g., '3-7') and returns a list of page numbers to process.
        If a single page number is provided (e.g., '4'), returns a list with that page.
        """
        pages = []
        if '-' in page_range:
            start, end = page_range.split('-')
            try:
                start, end = int(start), int(end)
                if start < 1 or end > total_pages:
                    raise ValueError(f"Page range must be between 1 and {total_pages}.")
                pages = list(range(start - 1, end))  # Adjust for 0-based index
            except ValueError as e:
                logger.error(f"Invalid page range: {page_range}. Error: {e}")
                raise
        else:
            try:
                page = int(page_range)
                if page < 1 or page > total_pages:
                    raise ValueError(f"Page number must be between 1 and {total_pages}.")
                pages = [page - 1]  # Adjust for 0-based index
            except ValueError as e:
                logger.error(f"Invalid page number: {page_range}. Error: {e}")
                raise
        return pages


    @staticmethod
    def _convert_pdf_to_images(pdf_path: str , temp_dir:str , pages:  List[int]) -> List[str]:
        """
        Converts a PDF file into a list of image file paths.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            os.makedirs(temp_dir, exist_ok=True)
            #images = convert_from_path(pdf_path, output_folder=temp_dir, fmt="jpeg")
            images = convert_from_path(pdf_path, first_page=pages[0] + 1, last_page=pages[-1] + 1, output_folder=temp_dir, fmt="jpeg")
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page-{i + 1}.jpg")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)
            return image_paths
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {pdf_path}", exc_info=True)
            raise
    def _cleanup_temp_dir(self, temp_dir: str):
        """
        Deletes the temporary directory and its contents.
        """
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory cleaned up: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {temp_dir}. Error: {e}")




    def convert_pdf_to_markdown(self, pdf_path: str = None ,prompt:str=None, pdf_url: str = None , pages_to_parse: str = None) -> str:
        """
        Converts a PDF file to Markdown format by processing its pages as images.

         Args:
        pdf_path (str, optional): Path to a local PDF file. Defaults to None.
        pdf_url (str, optional): URL to a PDF file. Defaults to None.
        pages_to_parse (str, optional): Range of pages to convert in format "start-end" 
                                        (e.g., "2-7" to convert pages 2 through 7). 
                                        If None, converts all pages. Defaults to None.
        prompt (str , optional): prompt: Instructions for how the pdf should be processed or described.
    
        Returns:
            str: Markdown representation of the specified PDF pages.

        """
        if not pdf_path and not pdf_url:
            raise ValueError("Either 'pdf_path' or 'pdf_url' must be provided.")

        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        temp_pdf_path = None

        try:
            if pdf_url:
                logger.info(f"Processing PDF from URL: {pdf_url}")
                temp_pdf_path = self._download_pdf_from_url(pdf_url, temp_dir)
            else:
                logger.info(f"Processing PDF from local path: {pdf_path}")
                temp_pdf_path = pdf_path

            # Get total number of pages in the PDF
            from PyPDF2 import PdfReader
            reader = PdfReader(temp_pdf_path)
            total_pages = len(reader.pages)

            # Parse the page range, defaulting to all pages if not provided
            if pages_to_parse:
                pages = self._parse_page_range(pages_to_parse, total_pages)
            else:
                pages = list(range(total_pages))  # Parse all pages if no range is given

            # Convert PDF pages to images
            image_paths = self._convert_pdf_to_images(temp_pdf_path ,temp_dir,pages)
            logger.info(f"Converted PDF to {len(image_paths)} images.")

            # Convert each image to Markdown
            markdown_parts = []
            for image_path in image_paths:
                logger.info(f"Processing image: {image_path}")
                markdown = self.client.convert_image_to_markdown(prompt=prompt,image_path=image_path)
                markdown_parts.append(markdown)

            # Combine Markdown parts from all pages
            combined_markdown = "\n\n".join(markdown_parts)
            logger.info("Successfully converted PDF to Markdown.")

            return combined_markdown

        except Exception as e:
            logger.error("Failed to convert PDF to Markdown", exc_info=True)
            raise
        finally:
            #Clean up temporary images 
            self._cleanup_temp_dir(temp_dir)