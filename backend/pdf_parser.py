import fitz  # PyMuPDF
import re
import os
import asyncio
import json
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class PDFLegalParser:
    def __init__(self, cap_number: str, pdf_dir: str = "."):
        self.cap_number = cap_number
        self.pdf_path = os.path.join(pdf_dir, f"cap{cap_number}.pdf")
        self.url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}!en.pdf?FROMCAPINDEX=Y"
        self.output_dir = "backend/data/parsed"
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def sort_cap_numbers(self, cap_list):
        """Sorts cap numbers like ['1', '207', '207A', '2'] into ['1', '2', '207', '207A']."""
        def key_func(cap):
            match = re.match(r"(\d+)([A-Z]*)", cap)
            if match:
                return (int(match.group(1)), match.group(2))
            return (0, cap)
        return sorted(cap_list, key=key_func)

    async def run_pipeline(self):
        """Main entry point for processing a single Ordinance."""
        print(f"\n=== Starting Pipeline for Cap {self.cap_number} ===")
        if not await self.download_pdf():
            print(f"Failed to download PDF for Cap {self.cap_number}")
            return []

        has_toc = await self.check_has_toc()
        
        if has_toc:
            print(f"TOC detected for Cap {self.cap_number}. Proceeding with structured parsing.")
            # Extract TOC text (first 40 pages as per plan)
            doc = fitz.open(self.pdf_path)
            toc_text = ""
            for i in range(min(40, len(doc))):
                toc_text += doc[i].get_text()
            doc.close()

            identified_list = await self._identify_sections_llm(toc_text)
            if not identified_list:
                print("No sections identified. Falling back to full text.")
                sections = await self.extract_full_text_fallback()
            else:
                structured_toc = await self._structure_toc_json_llm(identified_list, toc_text)
                sections = await self.extract_content_by_toc(structured_toc)
        else:
            print(f"No TOC detected for Cap {self.cap_number}. Using fallback extraction.")
            sections = await self.extract_full_text_fallback()

        if sections:
            self.save_to_json(sections)
        
        print(f"=== Completed Pipeline for Cap {self.cap_number} ===\n")
        return sections

    async def check_has_toc(self):
        """Uses LLM to determine if the PDF has a Table of Contents in the first 40 pages."""
        if not os.path.exists(self.pdf_path):
            return False
        
        doc = fitz.open(self.pdf_path)
        toc_text = ""
        for i in range(min(40, len(doc))):
            toc_text += doc[i].get_text()
        doc.close()

        if not toc_text.strip():
            return False

        prompt = f"Analyze this text from a Hong Kong Ordinance. Does it contain a 'Table of Contents' or 'Contents' section listing sections/schedules and page numbers? Respond with ONLY 'YES' or 'NO'.\n\nText snippet:\n{toc_text[:4000]}"
        
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            return "YES" in response.choices[0].message.content.upper()
        except Exception as e:
            print(f"TOC Detection failed: {e}")
            return False

    async def download_pdf(self, context=None):
        """Downloads the PDF from e-Legislation using Playwright to handle dynamic loading."""
        if os.path.exists(self.pdf_path):
            print(f"PDF for Cap {self.cap_number} already exists.")
            return True
        
        print(f"Downloading Cap {self.cap_number} using Playwright...")
        
        async def _download_with_context(ctx):
            page = await ctx.new_page()
            try:
                # If it's a new context, we might need to visit the home page once
                # but for simplicity let's just try direct download.
                # Actually, e-Legislation often needs a session.
                
                async with page.expect_download(timeout=60000) as download_info:
                    try:
                        await page.goto(self.url, wait_until="commit")
                    except Exception as e:
                        if "Download is starting" in str(e):
                            pass
                        else:
                            raise e
                
                download = await download_info.value
                await download.save_as(self.pdf_path)
                print(f"Download complete: {self.pdf_path}")
                return True
            except Exception as e:
                print(f"Failed to download PDF for Cap {self.cap_number}: {e}")
                return False
            finally:
                await page.close()

        if context:
            return await _download_with_context(context)
        else:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(accept_downloads=True)
                # Navigate to the main page first to establish session/cookies
                page = await context.new_page()
                await page.goto(f"https://www.elegislation.gov.hk/", wait_until="load", timeout=60000)
                await asyncio.sleep(2)
                await page.close()
                
                result = await _download_with_context(context)
                await browser.close()
                return result

    async def _identify_sections_llm(self, toc_text: str):
        """Step 1: Identify all sections and schedules from the TOC text as a plain list."""
        prompt = f"""
        You are a legal document parser. Your task is to scan the provided Table of Contents (TOC) text and list out EVERY section and schedule.
        
        Rules:
        - Include numeric sections (e.g., "1", "2", "18A").
        - Include Schedules (e.g., "First Schedule", "Second Schedule"). Note that Schedules are valid sections even if they don't have a simple number.
        - Ignore any sections marked as "(Repealed)" or "(Omitted)".
        - Ignore headers like "Section", "Page", "Last updated date", or Ordinance titles.
        - Output ONLY the list of section numbers/names, one per line.
        
        TOC Text:
        {toc_text}
        """
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a precise legal document parser. You output a clean list of sections found in the text."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Section Identification failed: {e}")
            return ""

    async def _structure_toc_json_llm(self, identified_list: str, toc_text: str):
        """Step 2: Match the identified list to titles and page labels and output JSON."""
        expected_sections = [line.strip() for line in identified_list.split('\n') if line.strip()]
        
        prompt = f"""
        You are a legal document parser. I have a list of sections identified from a Hong Kong Ordinance TOC.
        Your task is to find the "title" and "page_label" for each section in the list using the original TOC text.
        
        Identified Sections:
        {identified_list}
        
        Original TOC Text:
        {toc_text}
        
        Output Format:
        Return ONLY a JSON object with a key "sections" containing an array of objects.
        Each object must have:
        1. "section_no": The number (e.g., "5") or the full Schedule name (e.g., "First Schedule").
        2. "title": The descriptive title of the section.
        3. "page_label": The printed page label (e.g., "1-2", "S1-2", "3A-4").
        
        Example Output:
        {{
            "sections": [
                {{"section_no": "1", "title": "Short title", "page_label": "1-2"}},
                {{"section_no": "First Schedule", "title": "Specified Structures and Works", "page_label": "S1-2"}}
            ]
        }}
        """
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a precise legal document parser that outputs structured JSON based on a provided list and source text."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            sections = data.get("sections", [])
            
            # Verification and Correction: Ensure every section in identified_list is present and correctly named
            final_sections = []
            received_map = {str(s.get("section_no")).strip(): s for s in sections}
            
            missing = []
            for expected in expected_sections:
                # Try exact match, match without trailing dots, or case-insensitive match
                match = None
                if expected in received_map:
                    match = received_map[expected]
                elif (expected + ".") in received_map:
                    match = received_map[expected + "."]
                else:
                    # Case-insensitive search
                    for k, v in received_map.items():
                        if k.lower() == expected.lower() or k.lower() == (expected.lower() + "."):
                            match = v
                            break
                
                if match:
                    # Correction: Force the section_no to match the expected one exactly
                    match["section_no"] = expected
                    final_sections.append(match)
                else:
                    missing.append(expected)
            
            if missing:
                print(f"LLM missed {len(missing)} sections in JSON. Retrying for: {missing}")
                retry_prompt = f"""
                You previously missed some sections from the list. Please provide the "title" and "page_label" for ONLY these missing sections using the TOC text.
                
                Missing Sections:
                {", ".join(missing)}
                
                Original TOC Text:
                {toc_text}
                
                Output ONLY the JSON object with the "sections" key.
                """
                retry_response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a precise legal document parser. Complete the missing entries in the JSON."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                retry_data = json.loads(retry_response.choices[0].message.content)
                retry_sections = retry_data.get("sections", [])
                
                # Process retry results
                retry_map = {str(s.get("section_no")).strip(): s for s in retry_sections}
                for m in missing:
                    match = None
                    if m in retry_map:
                        match = retry_map[m]
                    elif (m + ".") in retry_map:
                        match = retry_map[m + "."]
                    else:
                        for k, v in retry_map.items():
                            if k.lower() == m.lower() or k.lower() == (m.lower() + "."):
                                match = v
                                break
                    
                    if match:
                        match["section_no"] = m
                        final_sections.append(match)
            
            return final_sections
        except Exception as e:
            print(f"LLM TOC structuring failed: {e}")
            return []

    async def _get_structured_toc_llm(self, toc_text: str):
        """Orchestrates the two-step LLM process to get a structured TOC."""
        print("Step 1: Identifying sections (Plain List)...")
        identified_list = await self._identify_sections_llm(toc_text)
        if not identified_list:
            print("No sections identified in Step 1.")
            return []
            
        print(f"Identified Sections:\n{identified_list}\n")
        print(f"The identified list has {len(identified_list.splitlines())} entries.")
        
        print("Step 2: Extracting details and structuring (JSON)...")
        return await self._structure_toc_json_llm(identified_list, toc_text)

    async def extract_full_text_fallback(self):
        """Fallback for PDFs without a TOC: Extract full text as a single chunk."""
        if not os.path.exists(self.pdf_path):
            return []
            
        doc = fitz.open(self.pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        clean_content = re.sub(r'\s+', ' ', full_text).strip()
        
        return [{
            "id": f"hk-cap{self.cap_number}-full",
            "content": clean_content,
            "title": f"Cap. {self.cap_number} (Full Text)",
            "citation": f"Cap. {self.cap_number}",
            "source_url": self.url,
            "page": "1",
            "type": "Ordinance"
        }]

    def save_to_json(self, sections):
        """Saves the parsed sections to a JSON file."""
        output_path = os.path.join(self.output_dir, f"cap{self.cap_number}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(sections)} sections to {output_path}")

    async def extract_content_by_toc(self, toc_entries):
        """
        Extracts content between the identified physical pages based on the structured TOC.
        """
        if not os.path.exists(self.pdf_path):
            return []

        doc = fitz.open(self.pdf_path)
        
        # 1. Build label_to_idx manually by scanning page text
        label_to_idx = {}
        for idx, page in enumerate(doc):
            text = page.get_text()
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in reversed(lines):
                if re.match(r'^[A-Z\d]+-[\d]+$', line):
                    if line not in label_to_idx:
                        label_to_idx[line] = idx
                    break

        # 2. Extract content
        sections = []
        # Add physical_idx to entries
        valid_entries = []
        for entry in toc_entries:
            p_idx = label_to_idx.get(entry["page_label"])
            if p_idx is not None:
                entry["physical_idx"] = p_idx
                valid_entries.append(entry)
            else:
                print(f"DEBUG: Dropped Section {entry['section_no']} - Page label '{entry['page_label']}' not found in PDF mapping.")
        
        valid_entries.sort(key=lambda x: x["physical_idx"])
        
        for i, entry in enumerate(valid_entries):
            start_idx = entry["physical_idx"]
            # End index is the start of the next section's page
            next_entry = valid_entries[i+1] if i+1 < len(valid_entries) else None
            end_idx = next_entry["physical_idx"] if next_entry else len(doc)
            
            # Ensure we always read at least the start page, even if next section is on the same page
            content = ""
            for p_idx in range(start_idx, max(start_idx + 1, end_idx)):
                content += doc[p_idx].get_text()
            
            # 1. Remove text BEFORE the current section (if multiple sections on same page)
            curr_no = entry["section_no"]
            if curr_no.isdigit() or (len(curr_no) > 1 and curr_no[:-1].isdigit()):
                curr_pattern = rf'(?:\n|^)\s*(?:Section\s+)?{re.escape(curr_no)}\.'
            else:
                curr_pattern = rf'(?:\n|^)\s*{re.escape(curr_no)}'
            
            curr_split = re.split(curr_pattern, content, flags=re.IGNORECASE)
            if len(curr_split) > 1:
                # Take everything after the last match of the current section header
                content = curr_split[-1]

            # 2. Remove text AFTER the next section starts
            if next_entry:
                next_no = next_entry["section_no"]
                if next_no.isdigit() or (len(next_no) > 1 and next_no[:-1].isdigit()):
                    next_pattern = rf'(?:\n|^)\s*(?:Section\s+)?{re.escape(next_no)}\.'
                else:
                    next_pattern = rf'(?:\n|^)\s*{re.escape(next_no)}'
                
                next_split = re.split(next_pattern, content, flags=re.IGNORECASE)
                content = next_split[0]

            # Final cleanup
            clean_content = re.sub(r'\s+', ' ', content).strip()
            
            if len(clean_content) > 10:
                sections.append({
                    "id": f"hk-cap{self.cap_number}-s{entry['section_no']}",
                    "content": clean_content,
                    "title": f"Cap. {self.cap_number} - {entry['title']}",
                    "citation": f"Cap. {self.cap_number}, s. {entry['section_no']}",
                    "source_url": f"{self.url}#page={start_idx + 1}",
                    "page": entry["page_label"],
                    "type": "Ordinance"
                })
            else:
                print(f"DEBUG: Dropped Section {entry['section_no']} - Content too short ({len(clean_content)} chars).")

        doc.close()
        return sections

if __name__ == "__main__":
    async def test():
        parser = PDFLegalParser()
        sections = await parser.run_pipeline()
        if sections:
            print(f"Successfully parsed {len(sections)} sections.")
            print("Sample Section:", sections[0]['citation'])
            print("Content Preview:", sections[0]['content'][:200])
    
    asyncio.run(test())
