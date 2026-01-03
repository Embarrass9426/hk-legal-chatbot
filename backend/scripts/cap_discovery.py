import asyncio
import json
import os
import re
from playwright.async_api import async_playwright

async def discover_caps():
    """
    Scrapes the e-Legislation numerical index to find all valid Cap numbers.
    """
    print("Starting Cap Discovery...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # The Chapter Number Index page as seen in the user's screenshot
        url = "https://www.elegislation.gov.hk/index/chapternumber?TYPE=1&TYPE=2&TYPE=3&LANGUAGE=E"
        
        try:
            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=90000)
            
            # Click "All" to show all entries if possible
            print("Attempting to show all entries...")
            try:
                # Based on the screenshot, there is an "All" link in the pagination area
                # We use a more specific selector to find the "All" link that shows all records
                all_link = page.locator("a:has-text('All')").last
                await all_link.click()
                await page.wait_for_load_state("networkidle")
                print("Clicked 'All' to show all entries.")
                # Wait a bit for the large table to render
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Could not click 'All', will scrape current page: {e}")

            # Extract Cap numbers from the first column of the table
            print("Extracting Cap numbers from table...")
            cap_numbers = await page.evaluate('''() => {
                // Find the table containing the index entries
                const rows = Array.from(document.querySelectorAll('table tr'));
                const caps = [];
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length > 0) {
                        const text = cells[0].innerText.trim();
                        // Match patterns like "1", "1A", "207", "207A"
                        if (/^[0-9]+[A-Z]*$/.test(text)) {
                            caps.push(text);
                        }
                    }
                });
                return caps;
            }''')
            
            if not cap_numbers:
                print("No Cap numbers found in table, trying regex fallback on content...")
                content = await page.content()
                cap_numbers = re.findall(r'Cap\.\s*([0-9]+[A-Z]*)', content)

            # Remove duplicates and sort
            unique_caps = sorted(list(set(cap_numbers)), key=lambda x: (int(re.match(r'(\d+)', x).group(1)), x))
            
            print(f"Found {len(unique_caps)} unique Caps.")
            
            # Save to JSON
            output_dir = "backend/data"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "cap_list.json")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(unique_caps, f, indent=2)
            
            print(f"Cap list saved to {output_path}")
            return unique_caps
            
        except Exception as e:
            print(f"Error during Cap discovery: {e}")
            # Fallback: If the selector fails, try to find any Cap. pattern in the whole page
            try:
                content = await page.content()
                cap_matches = re.findall(r'Cap\.\s*([0-9]+[A-Z]*)', content)
                unique_caps = sorted(list(set(cap_matches)), key=lambda x: (int(re.match(r'(\d+)', x).group(1)), x))
                if unique_caps:
                    print(f"Fallback: Found {len(unique_caps)} Caps using regex on full page.")
                    output_path = "backend/data/cap_list.json"
                    os.makedirs("backend/data", exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(unique_caps, f, indent=2)
                    return unique_caps
            except:
                pass
            return []
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(discover_caps())
