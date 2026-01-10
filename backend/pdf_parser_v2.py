import os
import re
import json
import warnings
import sys
from typing import List, Dict, Any

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*max_size parameter is deprecated.*")

def setup_environment():
    """Sets up PATH and DLL directories for GPU, Poppler, and Tesseract."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
    venv_site_packages = os.path.join(root_dir, ".venv", "Lib", "site-packages")
    
    # 1. Poppler setup
    poppler_path = os.path.join(current_file_dir, "bin", "poppler", "poppler-24.08.0", "Library", "bin")
    if not os.path.exists(poppler_path):
        poppler_path = os.path.join(root_dir, "backend", "bin", "poppler", "poppler-24.08.0", "Library", "bin")
    
    if os.path.exists(poppler_path):
        os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
        print(f"Added Poppler to PATH: {poppler_path}")

    # 3. GPU/CUDA/TensorRT DLL setup (Windows specific)
    if os.path.exists(venv_site_packages):
        gpu_dll_dirs = [
            os.path.join(venv_site_packages, "torch", "lib"),
            os.path.join(venv_site_packages, "nvidia", "cu13", "bin", "x86_64"),
            os.path.join(venv_site_packages, "nvidia", "cuda_runtime", "bin"),
            os.path.join(venv_site_packages, "tensorrt_libs")
        ]
        for dll_dir in gpu_dll_dirs:
            if os.path.exists(dll_dir):
                try:
                    os.add_dll_directory(os.path.abspath(dll_dir))
                    os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
                    print(f"Added DLL directory: {dll_dir}")
                except Exception as e:
                    print(f"Note: Could not add_dll_directory for {dll_dir}: {e}")

# Initialize environment before importing torch/unstructured
setup_environment()

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from transformers import AutoTokenizer
import torch

class PDFLegalParserV2:
    def __init__(self, cap_number: str, pdf_dir: str = "backend/data/pdfs"):
        self.cap_number = cap_number
        self.pdf_path = os.path.join(pdf_dir, f"cap{cap_number}.pdf")
        # Base URL for the PDF without query parameters for cleaner anchors
        self.base_pdf_url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}!en.pdf"
        self.url = f"{self.base_pdf_url}?FROMCAPINDEX=Y"
        self.output_dir = "backend/data/parsed"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tokenizer (matching Yuan embedding or generic BERT-based)
        self.tokenizer = AutoTokenizer.from_pretrained("IEITYuan/Yuan-embedding-2.0-en", trust_remote_code=True)
        
        # Elements to drop and keep
        self.DROP_TAGS = {
            "FigureCaption", "Header", "Footer", "PageBreak", 
            "Address", "EmailAddress", "CodeSnippet", "Formula", "Unknown"
        }
        self.KEEP_TAGS = {
            "Text", "Title", "NarrativeText", "ListItem", "Table", "Image", "UncategorizedText"
        }

    def parse_pdf(self, layout_batch_size=8):
        """Partitions PDF into elements using Unstructured."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF local path not found: {self.pdf_path}")

        # Verify GPU Availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Optimize for GPU and speed
        # Using yolox model and PaddleOCR for better compatibility/speed
        elements = partition_pdf(
            filename=self.pdf_path,
            strategy="hi_res",
            include_page_breaks=False,
            infer_table_structure=True,
            chunking_strategy=None,
            languages=["eng"],
            pdf_image_dpi=200,
            model_name="yolox",
            model_device="cuda" if cuda_available else "cpu",
            multiprocess=True,
            num_processes=8,
            layout_batch_size=layout_batch_size,
            ocr_languages=["eng"],
            pdf_extract_images=False,
            extract_image_block_output_dir=None,
        )
        return elements

    def is_title(self, el):
        """Robust check if element is a section title."""
        # Priority 1: Use unstructured's internal classification
        if el.category == "Title":
            return True
        
        # Priority 2: Heuristic: if text starts with "Section X" or "X." where X is a number, it's likely a title
        text = el.text.strip()
        if not text:
            return False

        # HK legislation headers often follow these patterns
        if re.match(r'^(Section|Schedule|Part|Chapter|Article|Instrument)\s+[A-Z\d]+', text, re.IGNORECASE):
            return True
        
        # Simple sequence numbering like "1." or "2."
        # Use a length heuristic to avoid matching long body paragraphs starting with numbers
        if re.match(r'^\d+\.', text) and len(text) < 150:
            return True
        
        return False

    def group_by_sections(self, elements):
        """Groups elements into sections using logic boundaries."""
        sections = []
        current_section = {
            "section_title": "Preamble",
            "elements": []
        }

        for el in elements:
            el_type = el.category
            
            if el_type in self.DROP_TAGS:
                continue
            
            # Special case for tables - always keep
            if el_type == "Table":
                current_section["elements"].append(el)
                continue

            if self.is_title(el):
                if current_section["elements"]:
                    sections.append(current_section)
                
                current_section = {
                    "section_title": el.text.strip(),
                    "elements": [el]
                }
            elif el_type in self.KEEP_TAGS:
                current_section["elements"].append(el)

        if current_section["elements"]:
            sections.append(current_section)
        
        return sections

    def normalize_section_content(self, elements):
        """Linearizes section elements into a single text block and tracks page boundaries."""
        full_text = ""
        page_mappings = [] # List of (char_index, page_number)
        
        current_char_idx = 0
        for el in elements:
            text = el.text.strip()
            if not text:
                continue
                
            page_num = el.metadata.page_number if hasattr(el.metadata, "page_number") else 1
            
            # Map the start of this element's text to its page number
            page_mappings.append((current_char_idx, page_num))
            
            full_text += text + "\n"
            current_char_idx += len(text) + 1 # +1 for the newline
            
        return full_text.strip(), page_mappings

    def get_page_for_offset(self, char_offset, page_mappings):
        """Finds the page number for a given character offset."""
        current_page = 1
        for offset, page in page_mappings:
            if char_offset >= offset:
                current_page = page
            else:
                break
        return current_page

    def chunk_section(self, section_data: Dict, chunk_size=300, overlap_percent=0.1):
        """Applies sliding-window token-based chunking within a section only."""
        content, page_mappings = self.normalize_section_content(section_data["elements"])
        
        if not content:
            return []

        # Tokenize with offsets to find where each chunk starts in the original text
        encoding = self.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        
        overlap = int(chunk_size * overlap_percent)
        chunks = []

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Get the page number for the start of this chunk
            start_char_idx = offsets[start][0]
            page_number = self.get_page_for_offset(start_char_idx, page_mappings)
            
            chunks.append({
                "doc_id": f"cap{self.cap_number}",
                "section_id": self._slugify(section_data["section_title"]),
                "section_title": section_data["section_title"],
                "content": chunk_text,
                "page_number": page_number,
                "chunk_index": chunk_idx,
                "citation": f"Cap. {self.cap_number}, {section_data['section_title']}",
                "source_url": f"{self.base_pdf_url}#page={page_number}"
            })
            
            chunk_idx += 1
            if end >= len(tokens):
                break
            start += (chunk_size - overlap)
                
        return chunks

    def _slugify(self, text):
        # Truncate to 100 chars to avoid Pinecone ID limit issues (512 total)
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        return slug[:100].strip('-')

    def process_ordinance(self, skip_if_exists=True, layout_batch_size=8):
        """Full pipeline for an ordinance. Skips parsing if JSON already exists."""
        output_path = os.path.join(self.output_dir, f"cap{self.cap_number}.json")
        
        if skip_if_exists and os.path.exists(output_path):
            print(f"Cap {self.cap_number} already parsed. Loading existing chunks from {output_path}...")
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        print(f"Parsing Cap {self.cap_number} with Layout Batch Size: {layout_batch_size}...")
        elements = self.parse_pdf(layout_batch_size=layout_batch_size)
        sections_elements = self.group_by_sections(elements)
        
        all_chunks = []
        for sec in sections_elements:
            sec_chunks = self.chunk_section(sec)
            all_chunks.extend(sec_chunks)
            
        # Add total_chunks_in_section metadata
        section_groups = {}
        for chunk in all_chunks:
            sid = chunk["section_id"]
            if sid not in section_groups:
                section_groups[sid] = 0
            section_groups[sid] += 1
            
        for chunk in all_chunks:
            chunk["total_chunks_in_section"] = section_groups[chunk["section_id"]]

        self.save_chunks(all_chunks)
        return all_chunks

    def save_chunks(self, chunks):
        output_path = os.path.join(self.output_dir, f"cap{self.cap_number}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = PDFLegalParserV2("282") # Example Cap 282
    try:
        chunks = parser.process_ordinance()
        print(f"Generated {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error: {e}")
