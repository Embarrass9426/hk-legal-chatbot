# PDF Parser Logic & Pseudocode

This document outlines the logic for the `PDFLegalParser` class to ensure consistency across the "all Caps" ingestion process.

## 🏗️ Class Structure: `PDFLegalParser`

### 1. `run_pipeline(cap_number)`
The main entry point for processing a single Ordinance.
1. `pdf_path = await download_pdf(cap_number)`
2. `has_toc = await check_has_toc(pdf_path)`
3. If `has_toc`:
    - `toc_text = extract_text(pdf_path, pages=1-40)`
    - `identified_list = await _identify_sections_llm(toc_text)`
    - `structured_toc = await _structure_toc_json_llm(identified_list, toc_text)`
    - `sections = extract_content_by_toc(pdf_path, structured_toc)`
4. Else:
    - `sections = extract_full_text_fallback(pdf_path)`
5. `save_to_json(sections, f"cap{cap_number}.json")`

---

## 🔍 Key Methods

### `check_has_toc(pdf_path)`
- **Input**: Path to PDF.
- **Logic**:
    - Extract text from the first 40 pages using PyMuPDF.
    - If text is empty, flag for OCR (Tesseract).
    - Send snippet to DeepSeek: "Does this text contain a Table of Contents listing sections and page numbers? Respond YES/NO."
- **Output**: Boolean.

### `_identify_sections_llm(toc_text)`
- **Logic**: LLM scans the TOC text and lists out EVERY section and schedule number/name.
- **Rules**: Include numeric sections (1, 2, 18A) and Schedules. Ignore repealed/omitted sections.
- **Output**: A plain text list of section numbers, one per line.

### `_structure_toc_json_llm(identified_list, toc_text)`
- **Logic**: LLM takes the plain list and the original TOC text to find the "Title" and "Page Label" for each entry.
- **Verification**: Includes a retry loop if the LLM misses sections from the `identified_list`.
- **Output**: List of objects `[{section_no, title, page_label}]`.

### `extract_content_by_toc(pdf_path, structured_toc)`
- **Logic**:
    1. **Map Labels**: Scan every page footer in the PDF to build a dictionary `{page_label: physical_index}`.
    2. **Sort**: Sort `structured_toc` by physical index.
    3. **Extract**: For each section:
        - `start_page = label_to_idx[current_section.page_label]`
        - `end_page = label_to_idx[next_section.page_label]` (or end of doc).
        - Extract text from `start_page` to `end_page`.
        - **Refine**: Use regex `Section {no}.` to find the exact start/end within those pages (handling multiple sections per page).
- **Output**: List of section dictionaries with content and metadata.

### `extract_full_text_fallback(pdf_path)`
- **Logic**:
    - Extract all text from the PDF.
    - If the document is short (< 5 pages), treat as one chunk.
    - If longer, attempt to split by "Section X" headers using regex.
- **Output**: List of section dictionaries.

---

## 💾 Storage Schema (`cap{num}.json`)
```json
[
  {
    "id": "hk-cap282-s5",
    "content": "...",
    "title": "Cap. 282 - Employer's liability for compensation",
    "citation": "Cap. 282, s. 5",
    "source_url": "https://www.elegislation.gov.hk/hk/cap282!en.pdf#page=12",
    "page": "12",
    "type": "Ordinance"
  }
]
```
