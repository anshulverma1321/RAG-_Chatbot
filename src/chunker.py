import os
import re
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Splits extracted text from document pages into paragraph-based chunks,
    preserving page numbers, paragraph numbers, and source document metadata.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the chunker.
        
        Args:
            chunk_size (int): Max character length of each chunk.
            chunk_overlap (int): Number of characters to overlap between chunks within a paragraph.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def split_pages(self, pages_data: list[dict]) -> list[dict]:
        """
        Splits text from each page into paragraphs, and then chunks each paragraph,
        preserving page numbers and assigning paragraph numbers.
        
        Args:
            pages_data (list[dict]): List of page dictionaries, containing keys:
                'text', 'page_number', 'source_document'.
                
        Returns:
            list[dict]: A list of chunks, each structured as:
                {
                    "text": "chunk text...",
                    "page_number": 3,
                    "paragraph_number": 2,
                    "chunk_id": "doc_filename_p3_para2_c0",
                    "source_document": "doc_filename.pdf"
                }
        """
        logger.info(f"Chunking pages with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        all_chunks = []
        
        for page_data in pages_data:
            text = page_data["text"]
            page_num = page_data["page_number"]
            doc_name = page_data["source_document"]
            
            if not text.strip():
                continue
                
            # Clean text: replace multiple consecutive newlines (3 or more) with double newlines
            text_clean = re.sub(r'\n{3,}', '\n\n', text)
            
            # Split page text into paragraphs
            raw_paragraphs = text_clean.split("\n\n")
            paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
            
            # Assign paragraph numbers (1-indexed)
            for para_idx, para_text in enumerate(paragraphs):
                para_num = para_idx + 1
                para_len = len(para_text)
                
                # If the paragraph is smaller than or equal to the chunk size, keep it as a single chunk
                if para_len <= self.chunk_size:
                    doc_slug = os.path.splitext(doc_name)[0].replace(" ", "_")
                    chunk_id = f"{doc_slug}_p{page_num}_para{para_num}_c0"
                    
                    all_chunks.append({
                        "text": para_text,
                        "page_number": page_num,
                        "paragraph_number": para_num,
                        "chunk_id": chunk_id,
                        "source_document": doc_name
                    })
                else:
                    # Otherwise, split the paragraph using a sliding window
                    start = 0
                    chunk_sub_idx = 0
                    
                    while start < para_len:
                        end = start + self.chunk_size
                        
                        if end >= para_len:
                            end = para_len
                            chunk_text = para_text[start:end]
                        else:
                            # Try to find a whitespace to avoid cutting words
                            split_pos = end
                            lookback_limit = max(start, end - min(self.chunk_overlap, 100))
                            
                            while split_pos > lookback_limit and not para_text[split_pos].isspace():
                                split_pos -= 1
                                
                            if split_pos > start:
                                end = split_pos
                                chunk_text = para_text[start:end]
                            else:
                                chunk_text = para_text[start:end]
                                
                        chunk_text_stripped = chunk_text.strip()
                        if chunk_text_stripped:
                            doc_slug = os.path.splitext(doc_name)[0].replace(" ", "_")
                            chunk_id = f"{doc_slug}_p{page_num}_para{para_num}_c{chunk_sub_idx}"
                            
                            all_chunks.append({
                                "text": chunk_text_stripped,
                                "page_number": page_num,
                                "paragraph_number": para_num,
                                "chunk_id": chunk_id,
                                "source_document": doc_name
                            })
                            chunk_sub_idx += 1
                            
                        if end == para_len:
                            break
                            
                        start = end - self.chunk_overlap
                        if start <= 0 or start >= end:
                            start = end
                            
        logger.info(f"Successfully chunked documents into {len(all_chunks)} chunks.")
        return all_chunks
