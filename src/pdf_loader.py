import os
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

class PDFLoader:
    """
    Loads and extracts text page-by-page from a PDF file.
    """
    def __init__(self, file_path: str):
        """
        Initializes the PDFLoader with a file path.
        
        Args:
            file_path (str): Path to the PDF file.
        """
        self.file_path = os.path.abspath(file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found at: {self.file_path}")
        
    def load(self) -> list[dict]:
        """
        Loads the PDF and extracts text from each page.
        
        Returns:
            list[dict]: A list of dictionaries containing text and page metadata:
                [
                    {
                        "text": "page content...",
                        "page_number": 1,
                        "source_document": "filename.pdf"
                    },
                    ...
                ]
        """
        logger.info(f"Attempting to open and parse PDF: {self.file_path}")
        
        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF file {self.file_path}: {e}")
            raise RuntimeError(f"Failed to read PDF file: {e}")
            
        pages_data = []
        filename = os.path.basename(self.file_path)
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Warn if page text is empty (e.g. scanned image PDFs)
                if not text.strip():
                    logger.warning(f"Page {page_num + 1} contains no extractable text.")
                    
                pages_data.append({
                    "text": text,
                    "page_number": page_num + 1,  # 1-indexed page number
                    "source_document": filename
                })
        except Exception as e:
            logger.error(f"Error occurred during text extraction from {filename}: {e}")
            raise RuntimeError(f"Error parsing PDF contents: {e}")
        finally:
            doc.close()
            
        logger.info(f"Successfully parsed {len(pages_data)} pages from {filename}")
        return pages_data
