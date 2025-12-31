from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time

def inspect_url(url):
    print(f"\n{'='*50}")
    print(f"Inspecting: {url}")
    print(f"{'='*50}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Navigate to URL
            print("Navigating...")
            page.goto(url, wait_until="networkidle")
            
            # Wait a bit for any extra JS to settle
            time.sleep(2)
            
            # Get content
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Print first 500 chars of body
            print("\n--- Body Preview (First 500 chars) ---")
            body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else "No body found"
            print(body_text[:500] + "...")
            
            # 1. Check for Title
            title = soup.title.string if soup.title else "No title found"
            print(f"\nPage Title: {title}")
            
            # 2. Identify Main Content Container
            print("\n--- Searching for buttons/tabs ---")
            buttons = page.query_selector_all("button")
            for i, btn in enumerate(buttons):
                text = btn.inner_text().strip()
                if text:
                    print(f"Button [{i}]: {text}")

            # 3. Look for Metadata
            print("\n--- Metadata Search ---")
            meta_elements = page.query_selector_all("div, span, td")
            for el in meta_elements:
                text = el.inner_text().lower()
                if any(kw in text for kw in ['citation:', 'date:', 'court:', 'cap:']):
                    print(f"Potential metadata found: {el.inner_text().strip()[:50]}")

        except Exception as e:
            print(f"Error inspecting {url}: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    samples = [
        "https://www.hklii.hk/en/legis/ord/1", # Interpretation and General Clauses Ordinance
    ]
    
    for url in samples:
        inspect_url(url)
