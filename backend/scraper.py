from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import asyncio
import re

class HKLIIScraper:
    def __init__(self):
        self.base_url = "https://www.hklii.hk"

    async def scrape_ordinance(self, cap_number: str):
        """
        Scrapes an ordinance from HKLII given its Cap number (e.g., '1', 'A1').
        """
        url = f"{self.base_url}/en/legis/ord/{cap_number}"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                print(f"Scraping {url}...")
                await page.goto(url, wait_until="networkidle")
                
                # Wait for content to load
                await asyncio.sleep(3)
                
                # Check if we need to click 'DISPLAY' or if content is already there
                # For now, let's just get the whole page content
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                title = soup.title.string if soup.title else f"Cap {cap_number}"
                
                # Extract sections
                sections = []
                
                # Try to find all divs with text
                main_content = await page.query_selector(".v-main__wrap")
                if main_content:
                    text = await main_content.inner_text()
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
                await browser.close()

if __name__ == "__main__":
    async def main():
        scraper = HKLIIScraper()
        results = await scraper.scrape_ordinance("1")
        print(f"Scraped {len(results)} sections.")
        if results:
            print("Sample section:", results[0])
    
    asyncio.run(main())
