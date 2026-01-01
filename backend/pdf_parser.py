import fitz  # PyMuPDF
import re
import os
import asyncio
from playwright.async_api import async_playwright

class PDFLegalParser:
    def __init__(self, cap_number: str):
        self.cap_number = cap_number
        self.pdf_path = f"cap{cap_number}.pdf"
        self.url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}!en.pdf"

    async def download_pdf(self):
        """Downloads the PDF from e-Legislation using Playwright to handle dynamic loading."""
        if os.path.exists(self.pdf_path):
            print(f"PDF for Cap {self.cap_number} already exists.")
            return True
        
        print(f"Downloading Cap {self.cap_number} using Playwright...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            
            try:
                # Navigate to the main page first to establish session/cookies
                print("Opening e-Legislation...")
                await page.goto(f"https://www.elegislation.gov.hk/", wait_until="load", timeout=60000)
                await asyncio.sleep(3)
                
                # Trigger download
                print(f"Requesting PDF for Cap {self.cap_number}...")
                async with page.expect_download(timeout=60000) as download_info:
                    try:
                        await page.goto(self.url, wait_until="commit")
                    except Exception as e:
                        if "Download is starting" in str(e):
                            print("Download started as expected.")
                        else:
                            raise e
                
                download = await download_info.value
                await download.save_as(self.pdf_path)
                print(f"Download complete: {self.pdf_path}")
                return True
            except Exception as e:
                print(f"Failed to download PDF with Playwright: {e}")
                return False
            finally:
                await browser.close()

    def parse_sections(self):
        """
        Parses the PDF by:
        1. Scanning all pages to build a map of printed page labels (e.g., "3A-10") to physical indices.
        2. Extracting section numbers, titles, and page labels from the TOC (pages 1-9).
        3. Extracting content between the identified physical pages.
        """
        if not os.path.exists(self.pdf_path):
            return []

        doc = fitz.open(self.pdf_path)
        
        # 1. Build label_to_idx manually by scanning page text (since doc.get_label() is empty)
        label_to_idx = {}
        for idx, page in enumerate(doc):
            text = page.get_text()
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            # Page labels like "1-12" or "3A-10" are usually at the bottom
            for line in reversed(lines):
                if re.match(r'^[A-Z\d]+-[\d]+$', line):
                    # We only take the first one we find from the bottom
                    if line not in label_to_idx:
                        label_to_idx[line] = idx
                    break

        # 2. Extract from TOC (Hardcoded Pages 1-9)
        toc_text = ""
        for i in range(min(9, len(doc))):
            toc_text += doc[i].get_text()
        
        toc_entries = []
        lines = [l.strip() for l in toc_text.split('\n') if l.strip()]
        
        i = 0
        while i < len(lines) - 1:
            # Match section number like "5." or "18A."
            if re.match(r'^\d+[A-Z]?\.$', lines[i]):
                section_no = lines[i][:-1]
                title_parts = []
                page_label = ""
                
                # Look ahead for the page label
                for j in range(i + 1, min(i + 10, len(lines))):
                    if re.match(r'^[A-Z\d]+-[\d]+$', lines[j]):
                        page_label = lines[j]
                        # Move outer index to after this entry
                        i = j 
                        break
                    else:
                        title_parts.append(lines[j])
                
                if page_label:
                    section_title = " ".join(title_parts)
                    
                    # Filter out repealed or omitted sections
                    title_lower = section_title.lower()
                    if "(repealed)" in title_lower or "(omitted)" in title_lower or "repealed" == title_lower or "omitted" == title_lower:
                        i += 1
                        continue

                    physical_idx = label_to_idx.get(page_label)
                    
                    if physical_idx is not None:
                        toc_entries.append({
                            "section_no": section_no,
                            "title": section_title,
                            "page_label": page_label,
                            "physical_idx": physical_idx
                        })
            i += 1

        # 3. Extract content
        sections = []
        toc_entries.sort(key=lambda x: x["physical_idx"])
        
        for i, entry in enumerate(toc_entries):
            start_idx = entry["physical_idx"]
            # End index is the start of the next section
            end_idx = toc_entries[i+1]["physical_idx"] if i+1 < len(toc_entries) else len(doc)
            
            if end_idx < start_idx:
                end_idx = start_idx + 1

            content = ""
            for p_idx in range(start_idx, end_idx):
                content += doc[p_idx].get_text()
            
            # Clean up: remove the next section's header if it's on the same page
            if i+1 < len(toc_entries):
                next_sec = toc_entries[i+1]
                # Look for "Section 6" or "6."
                next_header_pattern = rf'(?:Section\s+)?{next_sec["section_no"]}\.'
                split_content = re.split(next_header_pattern, content, flags=re.IGNORECASE)
                content = split_content[0]

            # Final cleanup
            clean_content = re.sub(r'\s+', ' ', content).strip()
            
            if len(clean_content) > 50:
                sections.append({
                    "id": f"hk-cap{self.cap_number}-s{entry['section_no']}",
                    "content": clean_content,
                    "title": f"Cap. {self.cap_number} - {entry['title']}",
                    "citation": f"Cap. {self.cap_number}, s. {entry['section_no']}",
                    "source_url": f"{self.url}#page={start_idx + 1}",
                    "page": entry["page_label"],
                    "type": "Ordinance"
                })

        doc.close()
        
        print("\n--- Parsed Sections ---")
        for s in sections:
            print(f"[{s['page']}] Section {s['id'].split('-s')[-1]}: {s['title']}")
        print(f"-----------------------\n")
        
        print(f"Parsed {len(sections)} sections using TOC-to-Page mapping.")
        return sections

if __name__ == "__main__":
    import asyncio
    async def test():
        parser = PDFLegalParser("282")
        if await parser.download_pdf():
            sections = parser.parse_sections()
            print(f"Parsed {len(sections)} sections.")
            if sections:
                print("Sample Section:", sections[0]['citation'])
                print("Content Preview:", sections[0]['content'][:200])
    
    asyncio.run(test())
