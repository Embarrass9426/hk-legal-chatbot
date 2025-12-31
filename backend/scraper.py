from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import re

class HKLIIScraper:
    def __init__(self):
        self.base_url = "https://www.hklii.hk"

    def scrape_ordinance(self, cap_number: str):
        """
        Scrapes an ordinance from HKLII given its Cap number (e.g., '1', 'A1').
        """
        url = f"{self.base_url}/en/legis/ord/{cap_number}"
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                print(f"Scraping {url}...")
                page.goto(url, wait_until="networkidle")
                
                # Wait for content to load
                time.sleep(3)
                
                # Check if we need to click 'DISPLAY' or if content is already there
                # For now, let's just get the whole page content
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                title = soup.title.string if soup.title else f"Cap {cap_number}"
                
                # Extract sections
                # This is a placeholder logic as HKLII structure is complex
                # We'll look for elements that look like sections
                sections = []
                
                # Try to find all divs with text
                main_content = page.query_selector(".v-main__wrap")
                if main_content:
                    text = main_content.inner_text()
                    # Simple chunking for now
                    chunks = text.split("\n\n")
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:
                            sections.append({
                                "id": f"hk-ord-cap{cap_number}-s{i}",
                                "title": title,
                                "content": chunk.strip(),
                                "source_url": url,
                                "citation": f"Cap. {cap_number}, Section {i}", # Placeholder
                                "type": "Ordinance"
                            })
                
                return sections

            except Exception as e:
                print(f"Error scraping {cap_number}: {e}")
                return []
            finally:
                browser.close()

if __name__ == "__main__":
    scraper = HKLIIScraper()
    results = scraper.scrape_ordinance("1")
    print(f"Scraped {len(results)} sections.")
    if results:
        print("Sample section:", results[0])
