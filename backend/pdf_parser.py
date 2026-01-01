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
        Parses the PDF and attempts to split it into sections, tracking page numbers.
        """
        if not os.path.exists(self.pdf_path):
            return []

        doc = fitz.open(self.pdf_path)
        sections = []
        
        # Pattern: Newline, then optional "Section ", then digits, then ". ", then Title
        pattern = r'(?:Section\s+)?(\d+[A-Z]?)\.\s+([A-Z][^\n]+)'
        
        current_section = None
        
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            
            # Find all section headers on this page
            matches = list(re.finditer(pattern, page_text))
            
            last_pos = 0
            for i, match in enumerate(matches):
                # If we were tracking a section, its content ends where this one starts
                if current_section:
                    current_section["content"] += page_text[last_pos:match.start()]
                    sections.append(current_section)
                
                section_no = match.group(1)
                section_title = match.group(2).strip()
                
                # Start a new section
                current_section = {
                    "id": f"hk-cap{self.cap_number}-s{section_no}",
                    "content": "", # Will be filled
                    "title": f"Cap. {self.cap_number} - {section_title}",
                    "citation": f"Cap. {self.cap_number}, s. {section_no}",
                    "source_url": f"{self.url}#page={page_num}",
                    "page": page_num,
                    "type": "Ordinance"
                }
                last_pos = match.start()
            
            # Add the rest of the page to the current section
            if current_section:
                current_section["content"] += page_text[last_pos:]
                last_pos = 0 # Reset for next page

        # Add the last section
        if current_section:
            sections.append(current_section)
            
        # Post-process: filter out short/invalid sections
        valid_sections = []
        for s in sections:
            if len(s["content"]) > 100 and "Contents" not in s["title"]:
                valid_sections.append(s)

        doc.close()
        return valid_sections

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
