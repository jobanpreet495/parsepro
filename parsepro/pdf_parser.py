import os
import logging
from typing import List
import tempfile
import requests
import shutil
from urllib.parse import urlparse
import asyncio




# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFToMarkdown:
    """
    A class to convert PDF files into Markdown format.
    """

    def __init__(self, provider="together", api_key=None):
        """
        Initializes the PDFToMarkdown client.
        
        Args:
            provider (str, optional): The API provider to use ('together', 'groq', or 'openai'). Defaults to "together".
            api_key (str, optional): The API key for the provider. If None, will try to get from environment.
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_env_key()
        
        if not self.api_key:
            env_var = f"{self.provider.upper()}_API_KEY"
            raise ValueError(f"API key is required. Set {env_var} environment variable or pass as an argument.")

        # Import here to avoid circular imports
        from parsepro.image_parser import ImageToMarkdown
        self.client = ImageToMarkdown(provider=self.provider, api_key=self.api_key)
        logger.info(f"PDFToMarkdown initialized with {self.provider.capitalize()} API client.")

    def _get_env_key(self):
        """
        Gets the API key from environment variables based on the provider.
        """
        env_keys = {
            "together": "TOGETHER_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY" ,
            "anthropic":"ANTHROPIC_API_KEY"
        }
        return os.getenv(env_keys.get(self.provider, ""))

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
    def _convert_pdf_to_images(pdf_path: str, temp_dir: str, pages: List[int]) -> List[str]:
        """
        Converts a PDF file into a list of image file paths.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            os.makedirs(temp_dir, exist_ok=True)
            from pdf2image import convert_from_path
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

    def convert_pdf_to_markdown(self, pdf_path: str = None, prompt: str = None, pdf_url: str = None, pages_to_parse: str = None, kwargs: dict = None) -> str:
        """
        Converts a PDF file to Markdown format by processing its pages as images.

        Args:
            pdf_path (str, optional): Path to a local PDF file. Defaults to None.
            pdf_url (str, optional): URL to a PDF file. Defaults to None.
            pages_to_parse (str, optional): Range of pages to convert in format "start-end" 
                                            (e.g., "2-7" to convert pages 2 through 7). 
                                            If None, converts all pages. Defaults to None.
            prompt (str, optional): Instructions for how the pdf should be processed or described.
            kwargs: Additional parameters including:
                    - json: Boolean to return JSON format (default: False)
                    - temperature: Controls randomness in the output (default: 0)
                    - kwargs={"json":False,"temperature":0}
        
        Returns:
            str or dict: Markdown text or JSON object with markdown field
        """
        if not pdf_path and not pdf_url:
            raise ValueError("Either 'pdf_path' or 'pdf_url' must be provided.")

        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        temp_pdf_path = None

        try:
            # If using Anthropic provider with a local PDF, use the specialized method
            if pdf_path  and self.provider == "anthropic":
                logger.info(f"Processing PDF with Anthropic API: {pdf_path}")
                markdown = self.convert_pdf(pdf_path = pdf_path)
                
                # Handle JSON output if requested
                if kwargs and kwargs.get("json") == True:
                    return {"markdown": markdown}
                return markdown
            
            elif pdf_url and self.provider=="anthropic":
                logger.info(f"Processing PDF with Anthropic API: {pdf_url}")
                markdown = self.convert_pdf(pdf_url=pdf_url)
                
                # Handle JSON output if requested
                if kwargs and kwargs.get("json") == True:
                    return {"markdown": markdown}
                return markdown
            


            elif pdf_url:
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
            image_paths = self._convert_pdf_to_images(temp_pdf_path, temp_dir, pages)
            logger.info(f"Converted PDF to {len(image_paths)} images.")

            # Convert each image to Markdown
            markdown_parts = []
            for image_path in image_paths:
                logger.info(f"Processing image: {image_path}")
                markdown = self.client.convert_image_to_markdown(image_path=image_path, prompt=prompt, kwargs=kwargs)
                if kwargs and kwargs.get("json") == True:
                    markdown = markdown["markdown"]

                markdown_parts.append(markdown)

            # Combine Markdown parts from all pages
            combined_markdown = "\n\n".join(markdown_parts)
            if kwargs and kwargs.get("json") == True:
                logger.info("Successfully converted PDF to Markdown.")
                return {"markdown": combined_markdown}
            else:
                return combined_markdown

        except Exception as e:
            logger.error("Failed to convert PDF to Markdown", exc_info=True)
            raise
        finally:
            # Clean up temporary images 
            self._cleanup_temp_dir(temp_dir)




    async def anthropic_pdf_to_markdown(self, pdf_path=None, pdf_url=None, max_concurrent=5):
        """
        Convert PDF to Markdown using Claude API with concurrent processing.
        
        Args:
            pdf_path: Path to the local PDF file (optional)
            pdf_url: URL to a PDF file (optional)
            max_concurrent: Maximum number of concurrent API calls
        
        Returns:
            Markdown string of the PDF content with pages in order
        
        Note:
            Either pdf_path or pdf_url must be provided, but not both.
        """
        import io
        import base64
        import asyncio
        import httpx
        from concurrent.futures import ThreadPoolExecutor
        from PyPDF2 import PdfReader, PdfWriter
        import anthropic
        
        # Validate input parameters
        if pdf_path is None and pdf_url is None:
            raise ValueError("Either pdf_path or pdf_url must be provided")
        if pdf_path is not None and pdf_url is not None:
            raise ValueError("Only one of pdf_path or pdf_url should be provided, not both")
        
        # Get PDF data
        if pdf_url is not None:
            # Download PDF from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(pdf_url)
                pdf_bytes = io.BytesIO(response.content)
            pdf = PdfReader(pdf_bytes)
        else:
            # Read local PDF file
            pdf = PdfReader(pdf_path)
        
        num_pages = len(pdf.pages)
        
        # Extract pages using a thread pool (since PDF operations are CPU-bound)
        def extract_page(page_num):
            writer = PdfWriter()
            writer.add_page(pdf.pages[page_num])
            
            page_bytes = io.BytesIO()
            writer.write(page_bytes)
            page_bytes.seek(0)
            
            return page_num, base64.standard_b64encode(page_bytes.read()).decode("utf-8")
        
        # Run extract_page for all pages in a thread pool
        loop = asyncio.get_event_loop()
        extracted_pages = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for page_num in range(num_pages):
                future = loop.run_in_executor(executor, extract_page, page_num)
                futures.append(future)
            
            # Wait for all futures to complete
            for future in await asyncio.gather(*futures):
                extracted_pages.append(future)
        
        # Process a single page with Claude API (this runs in a thread pool)
        def process_page(page_num, page_data):
            client = anthropic.Anthropic()
            
            try:
                # Make API call
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=2048,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": page_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": """
                                    Extract all content from this PDF page and convert it to clean, structured Markdown format.
                                    
                                    REQUIREMENTS:
                                    1. Return ONLY the extracted content in proper Markdown syntax
                                    2. Preserve the original document structure
                                    3. Do not include any explanatory text or meta-information
                                    """
                                }
                            ]
                        }
                    ],
                )
                
                page_markdown = message.content[0].text
                
                # Add page indicator if multi-page document
                if num_pages > 1:
                    page_markdown = f"\n\n## Page {page_num + 1}\n\n{page_markdown}"
                    
                return page_num, page_markdown
                
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                return page_num, f"\n\n## Page {page_num + 1}\n\n*Error processing this page*"
        
        # Process all pages with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_with_semaphore(page_num, page_data):
            async with semaphore:
                return await loop.run_in_executor(None, process_page, page_num, page_data)
        
        # Create and gather tasks
        tasks = []
        for page_num, page_data in extracted_pages:
            task = asyncio.create_task(process_with_semaphore(page_num, page_data))
            tasks.append(task)
        
        # Wait for all tasks to complete
        for result in await asyncio.gather(*tasks):
            results.append(result)
        
        # Sort results by page number and join
        results.sort(key=lambda x: x[0])
        all_markdown = [result[1] for result in results]
        
        return "\n\n".join(all_markdown)

    # Wrapper function for easier use
    def convert_pdf(self, pdf_path=None, pdf_url=None, max_concurrent=5):
        """
        Synchronous wrapper for the async function
        
        Args:
            pdf_path: Path to the local PDF file (optional)
            pdf_url: URL to a PDF file (optional)
            max_concurrent: Maximum number of concurrent API calls
        
        Returns:
            Markdown string of the PDF content
            
        Note:
            Either pdf_path or pdf_url must be provided, but not both.
        """
        return asyncio.run(self.anthropic_pdf_to_markdown(pdf_path=pdf_path, pdf_url=pdf_url, max_concurrent=max_concurrent))