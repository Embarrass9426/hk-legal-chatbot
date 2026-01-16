import os
import re
import json
import warnings
import sys
import site
from typing import List, Dict, Any
from pathlib import Path

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*max_size parameter is deprecated.*")

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# Fix for missing TensorRT DLLs (nvinfer_10.dll)
try:
    import tensorrt as trt
    trt_path = os.path.join(os.path.dirname(trt.__file__), "..", "tensorrt_libs")
    if os.path.exists(trt_path):
        os.environ["PATH"] = trt_path + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(trt_path)
except Exception:
    try:
        site_packages = site.getsitepackages()[0]
        trt_path = os.path.join(site_packages, "tensorrt_libs")
        if os.path.exists(trt_path):
            os.environ["PATH"] = trt_path + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(trt_path)
    except Exception:
        pass

class PDFLegalParserV2:
    def __init__(self, cap_number: str, pdf_dir: str = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\pdfs", output_dir: str = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\parsed", model=None, tokenizer=None):
        self.cap_number = cap_number
        self.pdf_path = os.path.join(pdf_dir, f"cap{cap_number}.pdf")
        # Base URL for the PDF without query parameters for cleaner anchors
        self.base_pdf_url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}!en.pdf"
        self.url = f"{self.base_pdf_url}?FROMCAPINDEX=Y"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model and tokenizer for semantic chunking
        model_name = "IEITYuan/Yuan-embedding-2.0-en"
        model_path = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\models\yuan-onnx-trt"
        self.device = "cpu" # Force CPU for torch to avoid sm_120 compatibility issues on RTX 50-series
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path if os.path.exists(model_path) else model_name, 
                trust_remote_code=True,
                fix_mistral_regex=True
            )
            
        if model:
            self.model = model
        elif os.path.exists(os.path.join(model_path, "model.onnx")):
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                provider=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[
                    {
                        "device_id": 0,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": model_path
                    },
                    {}, # CUDA
                    {}  # CPU
                ]
            )
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()

        
        # Elements to drop and keep
        self.DROP_TAGS = {
            "FigureCaption", "Header", "Footer", "PageBreak", 
            "Address", "EmailAddress", "CodeSnippet", "Formula", "Unknown"
        }
        self.KEEP_TAGS = {
            "Text", "Title", "NarrativeText", "ListItem", "Table", "Image", "UncategorizedText"
        }

    def _get_embeddings(self, texts: List[str]):
        """Helper to get semantic embeddings for a list of strings using CLS pooling."""
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Generate position_ids if required by the ORT model
            if "position_ids" not in inputs:
                batch_size, seq_len = inputs["input_ids"].shape
                inputs["position_ids"] = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Use CPU for tensors when calling ORT if device is CPU
            # Convert to numpy for ORT if using direct InferenceSession, 
            # but ORTModelForFeatureExtraction handles torch tensors if they are on the right device.
            outputs = self.model(**inputs)
            
            # Use CLS pooling for Yuan-embedding (Index 0)
            # Check the output type - ORTModelForFeatureExtraction usually returns a ModelOutput
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                # Handle tuple/dict output from ORT
                # Some versions/configs return a tuple where index 0 is the hidden states
                # Let's be safer and check if it's a tensor first
                hidden_states = outputs[0] if isinstance(outputs, (list, tuple)) else outputs["last_hidden_state"]
                embeddings = hidden_states[:, 0, :]
                
            return F.normalize(embeddings, p=2, dim=1)

    def _count_tokens(self, text: str):
        """Counts tokens using the class tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def parse_pdf(self, layout_batch_size=8):
        """Partitions PDF into elements using Unstructured 'fast' strategy."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF local path not found: {self.pdf_path}")

        print(f"Parsing Cap {self.cap_number} with 'fast' strategy...")
        elements = partition_pdf(
            filename=self.pdf_path,
            strategy="fast",
            include_page_breaks=False,
            languages=["eng"],
            infer_table_structure=True,
            layout_batch_size=layout_batch_size
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
    
    # First initialize sections
    # Then group elements into sections based on titles
    # Finally return list of sections with their elements
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
    
    # To record page boundaries by mapping char offsets to page numbers while linearizing
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
            
            # Join elements with double newlines to clearly separate paragraphs
            full_text += text + "\n\n"
            current_char_idx += len(text) + 2 # +2 for the double newline
            
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

    def chunk_section_paragraph_based(self, section_data: Dict, chunk_size=1200, overlap_sentences=2, threshold=0.8):
        """
        Semantic chunking based on paragraphs. Merges paragraphs which are semantically 
        similar until the chunk_size (upper limit) is reached.
        """
        # Step 1: Convert PDF section elements into a single text block
        content, page_mappings = self.normalize_section_content(section_data["elements"])

        if not content:
            return []

        # Step 2: Split text into paragraphs (separated by double newlines)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        # Step 3: Get embeddings for semantic similarity
        paragraph_embeddings = self._get_embeddings(paragraphs)

        chunks = []                # will store final chunks
        current_paragraphs = []    # buffer of paragraphs in current chunk
        current_tokens = 0
        chunk_index = 0

        # Trace paragraph offsets to map back to page numbers
        paragraph_offsets = []
        current_offset = 0
        for p in paragraphs:
            paragraph_offsets.append(current_offset)
            current_offset += len(p) + 2 # +2 for the \n\n joins

        for i in range(len(paragraphs)):
            paragraph = paragraphs[i]
            paragraph_tokens = self._count_tokens(paragraph)

            # determine similarity to previous paragraph (if any)
            if i > 0:
                similarity = F.cosine_similarity(
                    paragraph_embeddings[i-1].unsqueeze(0),
                    paragraph_embeddings[i].unsqueeze(0)
                ).item()
            else:
                similarity = 1.0

            # merge paragraphs if semantically close and not exceeding chunk size (upper limit)
            if similarity >= threshold and (current_tokens + paragraph_tokens) <= chunk_size:
                current_paragraphs.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                # flush current chunk
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    # Page mapping is based on the start of the first paragraph in the chunk
                    first_para_idx = i - len(current_paragraphs)
                    start_char_idx = paragraph_offsets[first_para_idx]
                    page_number = self.get_page_for_offset(start_char_idx, page_mappings)

                    chunks.append({
                        "content": chunk_text,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "section_id": self._slugify(section_data["section_title"]),
                        "section_title": section_data["section_title"],
                        "doc_id": "cap" + self.cap_number,
                        "citation": f"Cap. {self.cap_number}, {section_data['section_title']}",
                        "source_url": f"{self.base_pdf_url}#page={page_number}"
                    })
                    chunk_index += 1

                # start a new chunk buffer
                current_paragraphs = [paragraph]
                current_tokens = paragraph_tokens

        # flush last chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            first_para_idx = len(paragraphs) - len(current_paragraphs)
            start_char_idx = paragraph_offsets[first_para_idx]
            page_number = self.get_page_for_offset(start_char_idx, page_mappings)

            chunks.append({
                "content": chunk_text,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "section_id": self._slugify(section_data["section_title"]),
                "section_title": section_data["section_title"],
                "doc_id": "cap" + self.cap_number,
                "citation": f"Cap. {self.cap_number}, {section_data['section_title']}",
                "source_url": f"{self.base_pdf_url}#page={page_number}"
            })

        # Step 4: Add sentence-level overlap
        overlapped_chunks = []

        for j in range(len(chunks)):
            current_chunk_data = chunks[j].copy()
            current_chunk_text = current_chunk_data["content"]

            # copy last N sentences from previous chunk to current
            if j > 0 and overlap_sentences > 0:
                prev_chunk_text = chunks[j-1]["content"]
                
                # Split previous chunk into sentences using regex
                prev_sentences = re.findall(r'[^.!?]+[.!?]+', prev_chunk_text)
                if not prev_sentences:
                    prev_sentences = [prev_chunk_text.strip()]
                
                overlap_text = " ".join(prev_sentences[-overlap_sentences:])
                
                # prepend overlap to the current chunk
                current_chunk_text = overlap_text + " " + current_chunk_text
            
            # store adjusted chunk with new content
            current_chunk_data["content"] = current_chunk_text
            overlapped_chunks.append(current_chunk_data)

        return overlapped_chunks


    def _slugify(self, text):
        # Truncate to 100 chars to avoid Pinecone ID limit issues (512 total)
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        return slug[:100].strip('-')

    def process_ordinance(self, skip_if_exists=True, layout_batch_size=16):
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
            sec_chunks = self.chunk_section_paragraph_based(sec)
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
