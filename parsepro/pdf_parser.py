import os
import logging
from typing import List
import tempfile
import requests
import shutil
from urllib.parse import urlparse

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
                markdown = self.convert_pdf(pdf_path = pdf_path,kwargs=kwargs, page_range=pages_to_parse)
                
                # Handle JSON output if requested
                if kwargs and kwargs.get("json") == True:
                    return {"markdown": markdown}
                return markdown
            
            elif pdf_url and self.provider=="anthropic":
                logger.info(f"Processing PDF with Anthropic API: {pdf_url}")
                markdown = self.convert_pdf(pdf_url=pdf_url,kwargs=kwargs ,page_range=pages_to_parse)
                
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


    async def anthropic_pdf_to_markdown(self, pdf_path=None, pdf_url=None, max_concurrent=5, page_range=None, kwargs=None):
        """
        Convert PDF to Markdown using Claude API with concurrent processing.
        
        Args:
            pdf_path: Path to the local PDF file (optional)
            pdf_url: URL to a PDF file (optional)
            max_concurrent: Maximum number of concurrent API calls
            page_range: Range of pages to process (e.g., "1-5", "3-7") (optional)
                        Page numbering starts at 1 to match PDF viewers
        
        Returns:
            Markdown string of the PDF content with pages in order
        
        Note:
            Either pdf_path or pdf_url must be provided, but not both.
        """
        import io
        import base64
        import asyncio
        import httpx
        import logging
        import time
        from concurrent.futures import ThreadPoolExecutor
        from PyPDF2 import PdfReader, PdfWriter
        import anthropic
        
        # Set up detailed logging
        logging.info(f"Starting PDF to Markdown conversion with parameters: pdf_path={pdf_path}, pdf_url={pdf_url}, max_concurrent={max_concurrent}, page_range={page_range}")
        
        # Validate input parameters
        if pdf_path is None and pdf_url is None:
            raise ValueError("Either pdf_path or pdf_url must be provided")
        if pdf_path is not None and pdf_url is not None:
            raise ValueError("Only one of pdf_path or pdf_url should be provided, not both")
        
        # Get PDF data
        if pdf_url is not None:
            # Download PDF from URL with retries and extended timeout
            max_retries = 3
            timeout = httpx.Timeout(60.0, connect=30.0)  # 60 seconds total, 30 seconds for connection
            
            for retry in range(max_retries):
                try:
                    logging.info(f"Attempting to download PDF from {pdf_url} (attempt {retry+1}/{max_retries})")
                    start_time = time.time()
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(pdf_url)
                        download_time = time.time() - start_time
                        logging.info(f"Download took {download_time:.2f} seconds")
                        
                        if response.status_code == 200:
                            pdf_bytes = io.BytesIO(response.content)
                            pdf_size = len(response.content)
                            logging.info(f"Successfully downloaded PDF from {pdf_url} (size: {pdf_size/1024:.2f} KB)")
                            break
                        else:
                            logging.error(f"Failed to download PDF. Status code: {response.status_code}")
                            raise ValueError(f"Failed to download PDF. Status code: {response.status_code}")
                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry  # Exponential backoff
                        logging.warning(f"Connection error: {str(e)}. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"Failed to download PDF after {max_retries} attempts: {str(e)}")
                        raise ValueError(f"Failed to download PDF after {max_retries} attempts: {str(e)}")
            
            try:
                logging.info("Parsing downloaded PDF...")
                start_time = time.time()
                pdf = PdfReader(pdf_bytes)
                parsing_time = time.time() - start_time
                logging.info(f"PDF parsing took {parsing_time:.2f} seconds")
            except Exception as e:
                logging.error(f"Error parsing PDF: {str(e)}")
                raise ValueError(f"Error parsing PDF: {str(e)}")
        else:
            # Read local PDF file
            try:
                logging.info(f"Reading PDF from local path: {pdf_path}")
                start_time = time.time()
                pdf = PdfReader(pdf_path)
                parsing_time = time.time() - start_time
                logging.info(f"PDF parsing took {parsing_time:.2f} seconds")
            except Exception as e:
                logging.error(f"Error reading PDF from path {pdf_path}: {str(e)}")
                raise ValueError(f"Error reading PDF from path {pdf_path}: {str(e)}")
        
        num_pages = len(pdf.pages)
        logging.info(f"PDF loaded successfully with {num_pages} pages")
        
        # Parse page range if provided
        pages_to_process = list(range(num_pages))
        if page_range:
            try:
                # Parse the range (e.g., "3-5")
                start, end = map(int, page_range.split('-'))
                # Convert from 1-indexed (user-friendly) to 0-indexed (internal)
                start = max(0, start - 1)  # Ensure start is not negative
                end = min(num_pages, end)  # Ensure end is not beyond last page
                pages_to_process = list(range(start, end))
                
                if not pages_to_process:
                    raise ValueError(f"Invalid page range: {page_range}. PDF has {num_pages} pages.")
            except ValueError as e:
                if "Invalid page range" in str(e):
                    raise
                raise ValueError(f"Invalid page range format: {page_range}. Use format like '1-5'")
        
        logging.info(f"Processing pages: {[p+1 for p in pages_to_process]}")
        
        # Extract pages using a thread pool (since PDF operations are CPU-bound)
        def extract_page(page_num):
            try:
                logging.info(f"Extracting page {page_num+1}...")
                start_time = time.time()
                
                writer = PdfWriter()
                writer.add_page(pdf.pages[page_num])
                
                page_bytes = io.BytesIO()
                writer.write(page_bytes)
                page_bytes.seek(0)
                
                page_data = base64.standard_b64encode(page_bytes.read()).decode("utf-8")
                extraction_time = time.time() - start_time
                logging.info(f"Page {page_num+1} extraction took {extraction_time:.2f} seconds")
                
                return page_num, page_data
            except Exception as e:
                logging.error(f"Error extracting page {page_num+1}: {str(e)}")
                return page_num, None
        
        # Run extract_page for specified pages in a thread pool
        logging.info("Starting page extraction...")
        loop = asyncio.get_event_loop()
        extracted_pages = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for page_num in pages_to_process:
                future = loop.run_in_executor(executor, extract_page, page_num)
                futures.append(future)
            
            # Wait for all futures to complete
            for future in await asyncio.gather(*futures):
                if future[1] is not None:  # Only add successful extractions
                    extracted_pages.append(future)
        
        if not extracted_pages:
            raise ValueError("Failed to extract any pages from the PDF")
        
        logging.info(f"Successfully extracted {len(extracted_pages)} pages")
        
        # Process a single page with Claude API (this runs in a thread pool)
        def process_page(page_num, page_data):
            logging.info(f"Processing page {page_num+1} with Claude API...")
            start_time = time.time()
            
            client = anthropic.Anthropic()
            
            if kwargs is not None:
                temperature = kwargs.get("temperature", 0)
            else:
                temperature = 0
            
            try:
                # Make API call with timeout
                message = client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4096,
                    temperature=temperature,
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
                processing_time = time.time() - start_time
                logging.info(f"Page {page_num+1} processing took {processing_time:.2f} seconds")
                
                # Add page indicator if multi-page document
                if len(pages_to_process) > 1:
                    # Display the actual PDF page number (1-indexed) to the user
                    page_markdown = f"\n\n## Page {page_num + 1}\n\n{page_markdown}"
                    
                return page_num, page_markdown
                
            except Exception as e:
                logging.error(f"Error processing page {page_num+1} with Claude API: {str(e)}")
                return page_num, f"\n\n## Page {page_num + 1}\n\n*Error processing this page: {str(e)}*"
        
        # Process all pages with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_with_semaphore(page_num, page_data):
            async with semaphore:
                try:
                    # Add a timeout to prevent hanging
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, process_page, page_num, page_data),
                        timeout=120  # 2 minutes timeout per page
                    )
                except asyncio.TimeoutError:
                    logging.error(f"Timeout while processing page {page_num+1}")
                    return page_num, f"\n\n## Page {page_num + 1}\n\n*Error: Processing timed out after 120 seconds*"
        
        # Create and gather tasks
        logging.info("Starting Claude API processing...")
        tasks = []
        for page_num, page_data in extracted_pages:
            task = asyncio.create_task(process_with_semaphore(page_num, page_data))
            tasks.append(task)
        
        # Wait for all tasks to complete with overall timeout
        try:
            results_gathered = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=300  # 5 minutes overall timeout
            )
            for result in results_gathered:
                results.append(result)
        except asyncio.TimeoutError:
            logging.error("Overall timeout exceeded while processing pages")
            # Add partial results if available
            for task in tasks:
                if task.done():
                    try:
                        results.append(task.result())
                    except Exception:
                        pass  # Ignore errors in completed tasks
        
        # Sort results by page number and join
        results.sort(key=lambda x: x[0])
        all_markdown = [result[1] for result in results]
        
        logging.info(f"Successfully processed {len(results)} pages with Claude API")
        
        return "\n\n".join(all_markdown)
    # Wrapper function for easier use
    def convert_pdf(self, pdf_path=None, pdf_url=None, max_concurrent=5, page_range=None, kwargs=None):
        """
        Synchronous wrapper for the async function
        
        Args:
            pdf_path: Path to the local PDF file (optional)
            pdf_url: URL to a PDF file (optional)
            max_concurrent: Maximum number of concurrent API calls
            page_range: Range of pages to process (e.g., "1-5", "3-7") (optional)
                        Page numbering starts at 1 to match PDF viewers
        
        Returns:
            Markdown string of the PDF content
            
        Note:
            Either pdf_path or pdf_url must be provided, but not both.
        """
        import asyncio
        import logging
        import time
        
        start_time = time.time()
        logging.info(f"Starting PDF conversion: pdf_path={pdf_path}, pdf_url={pdf_url}, page_range={page_range}")
        
        try:
            # Create a new event loop to ensure a clean state
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.anthropic_pdf_to_markdown(
                    pdf_path=pdf_path, 
                    pdf_url=pdf_url, 
                    max_concurrent=max_concurrent,
                    page_range=page_range,
                    kwargs=kwargs
                )
            )
            
            total_time = time.time() - start_time
            logging.info(f"PDF conversion completed in {total_time:.2f} seconds")
            
            return result
        except ValueError as e:
            logging.error(f"Error in PDF conversion: {str(e)}")
            return f"## Error converting PDF\n\n{str(e)}"
        except asyncio.TimeoutError:
            logging.error("PDF conversion timed out")
            return "## Error converting PDF\n\nThe operation timed out. Please try with a smaller page range or a different PDF."
        except Exception as e:
            logging.error(f"Unexpected error in PDF conversion: {str(e)}")
            return f"## Unexpected error\n\n{str(e)}"
        finally:
            # Clean up the event loop
            try:
                loop.close()
            except:
                pass