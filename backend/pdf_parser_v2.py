# backend/pdf_parser_v2.py
from email.mime import text
import os, re, json, time, warnings, torch, torch.nn.functional as F
from typing import Dict
from unstructured.partition.pdf import partition_pdf
from embedding_shared import job_q, result_q

class PDFLegalParserV2:
    def __init__(self, cap_number: str, pdf_dir: str, output_dir: str):
        self.cap_number = cap_number
        self.pdf_path = os.path.join(pdf_dir, f"cap{cap_number}.pdf")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_pdf_url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}!en.pdf"
        self.url = f"{self.base_pdf_url}?FROMCAPINDEX=Y"
        # Elements to drop entirely (noise / non‑content)
        self.DROP_TAGS = {
            "Header",
            "Footer",
            "PageHeader",
            "PageFooter",
            "FigureCaption",
            "Figure",
            "Picture",
            "Image",
            "Attachment",
            "Formula",
            "Equation",
            "TableCaption",
            "PageNumber",
            "Metadata",
            "TextBox",
            "SectionHeader",
            "Footnote",
            "Endnote",
            "Citation",
            "NumberedFigure",
            "UncategorizedText",
        }

        # Elements we consider meaningful text content
        self.KEEP_TAGS = {
            "Title",
            "ListItem",
            "NarrativeText",
            "Table",
            "Body",
            "Paragraph",
        }

    # ──────────────────────────────────────────────
    # Get embeddings via queue (never run model directly)
    # ──────────────────────────────────────────────
    def _get_embeddings(self, texts):
        import uuid, time
        job_id = str(uuid.uuid4())
        job_q.put({"type": "embed_request", "id": job_id, "texts": texts})

        while True:
            result = result_q.get()
            if result.get("id") == job_id:
                if "error" in result:
                    raise RuntimeError(f"Embedding worker error: {result['error']}")
                return result["vectors"]
            else:
                result_q.put(result)
                time.sleep(0.01)

    # ──────────────────────────────────────────────
    # PDF parsing & chunking (unchanged logic)
    # ──────────────────────────────────────────────
    def parse_pdf(self, layout_batch_size=128):
        elements = partition_pdf(
            filename=self.pdf_path,
            strategy="fast",
            include_page_breaks=False,
            languages=["eng"],
            infer_table_structure=True,
            layout_batch_size=layout_batch_size
        )
        return elements

    def load_parsed_json(self, path: str):
            """Load previously parsed chunks JSON from disk."""
            if not os.path.exists(path):
                raise FileNotFoundError(f"Parsed JSON file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📄 Loaded {len(data)} chunks from {path}")
            return data

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
    
    def _count_tokens(self, text: str) -> int:
        """Simple whitespace-based token counting."""
        return len(text.split())

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

        # 🔧 Convert to tensor if worker returned NumPy arrays
        if not torch.is_tensor(paragraph_embeddings):
            paragraph_embeddings = torch.tensor(paragraph_embeddings, dtype=torch.float32)

        chunks = []
        current_paragraphs = []
        current_tokens = 0
        chunk_index = 1

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

    def process_ordinance(self, skip_if_exists=True, layout_batch_size=128):
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
            sec_chunks = self.chunk_section_paragraph_based(sec, overlap_sentences=2)
            
            # Filter out chunks where content matches section_id exactly
            # These are usually just redundant headers or structure elements
            filtered_sec_chunks = []
            for chunk in sec_chunks:
                content_stripped = chunk["content"].strip().lower()
                id_stripped = chunk["section_id"].strip().lower()
                title_stripped = chunk["section_title"].strip().lower()
                
                # Drop if content is exactly the same as section_id or section_title
                if content_stripped == id_stripped or content_stripped == title_stripped or not content_stripped:
                    continue
                filtered_sec_chunks.append(chunk)
            
            # Re-index remaining chunks for this section (1-based as per user example)
            for i, chunk in enumerate(filtered_sec_chunks, 1):
                chunk["chunk_index"] = i
                
            all_chunks.extend(filtered_sec_chunks)
            
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
    from pathlib import Path

    pdf_dir = Path(r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\pdfs")
    parsed_dir = Path(r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\backend\data\parsed")

    parser = PDFLegalParserV2("282", pdf_dir=pdf_dir, output_dir=parsed_dir)
    chunks = parser.process_ordinance(skip_if_exists=False)
    print(f"✅ Parsed {len(chunks)} chunks for Cap 282 (without embeddings).")

