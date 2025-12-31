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
        Parses the PDF and attempts to split it into sections.
        """
        if not os.path.exists(self.pdf_path):
            return []

        doc = fitz.open(self.pdf_path)
        full_text = ""
        for page in doc:
            # Filter out headers/footers if possible, or just get all text
            full_text += page.get_text()

        # HK Ordinances usually have sections starting with a number followed by a title
        # or "Section X". 
        # We'll look for patterns like "\n5. Interpretation" or "\nSection 5"
        sections = []
        
        # Split by "Section X" or "X. " at the start of a line
        # We use a more complex regex to avoid matching numbers inside text
        # Pattern: Newline, then optional "Section ", then digits, then ". ", then Title
        pattern = r'\n(?:Section\s+)?(\d+[A-Z]?)\.\s+([A-Z][^\n]+)'
        
        matches = list(re.finditer(pattern, full_text))
        
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            
            section_no = matches[i].group(1)
            section_title = matches[i].group(2).strip()
            content = full_text[start:end].strip()
            
            # Skip very short chunks or table of contents
            if len(content) < 100 or "Contents" in section_title:
                continue

            sections.append({
                "id": f"hk-cap{self.cap_number}-s{section_no}",
                "content": content,
                "title": f"Cap. {self.cap_number} - {section_title}",
                "citation": f"Cap. {self.cap_number}, s. {section_no}",
                "source_url": self.url,
                "type": "Ordinance"
            })
            
        doc.close()
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
