import asyncio
import json
import os
import argparse
from playwright.async_api import async_playwright
import sys

# Add the parent directory to sys.path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.pdf_parser import PDFLegalParser

async def batch_download(start_page=1, end_page=158, concurrency=5):
    # Paths
    pdf_dir = os.path.join("backend", "data", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        
        # Initialize session
        print("Visiting home page...")
        page = await context.new_page()
        await page.goto("https://www.elegislation.gov.hk/", wait_until="load", timeout=60000)
        await asyncio.sleep(5)
        
        semaphore = asyncio.Semaphore(concurrency)
        failed_caps = []

        async def download_one(cap, href):
            async with semaphore:
                parser = PDFLegalParser(cap, pdf_dir=pdf_dir)
                if os.path.exists(parser.pdf_path):
                    print(f"  Cap {cap} already exists.")
                    return True
                
                download_page = await context.new_page()
                success = False
                try:
                    url = f"https://www.elegislation.gov.hk{href}"
                    if "FROMCAPINDEX" not in url:
                         url += "?FROMCAPINDEX=Y"

                    async with download_page.expect_download(timeout=30000) as download_info:
                        try:
                            await download_page.goto(url, wait_until="commit")
                        except Exception as e:
                            if "Download is starting" in str(e):
                                pass
                            else:
                                raise e
                    download = await download_info.value
                    await download.save_as(parser.pdf_path)
                    print(f"  Saved Cap {cap}")
                    success = True
                except Exception as e:
                    failed_caps.append(cap)
                    if "Timeout" in str(e):
                         print(f"  Skipping Cap {cap} (Timeout/Not a PDF link)")
                    else:
                        print(f"  Failed to download Cap {cap}: {e}")
                finally:
                    await download_page.close()
                return success

        current_page_num = start_page
        while True:
            if end_page and current_page_num > end_page:
                break
                
            index_url = f"https://www.elegislation.gov.hk/index/chapternumber?TYPE=1&TYPE=2&TYPE=3&LANGUAGE=E&p0={current_page_num}"
            print(f"\nScanning Page {current_page_num}: {index_url}")
            
            await page.goto(index_url, wait_until="networkidle")
            await asyncio.sleep(2) # Wait for table to render
            
            # Find rows
            rows = await page.query_selector_all("tr.even, tr.odd")
            tasks = []
            
            for row in rows:
                cols = await row.query_selector_all("td")
                if len(cols) < 4:
                    continue
                
                cap_text = (await cols[0].inner_text()).strip()
                if not cap_text:
                    continue
                
                # Check for English PDF link (4th column usually)
                eng_col = cols[3]
                pdf_link = await eng_col.query_selector("a[href*='.pdf']")
                target_href = None
                
                if pdf_link:
                    target_href = await pdf_link.get_attribute("href")
                
                if target_href:
                    tasks.append(download_one(cap_text, target_href))
            
            if tasks:
                print(f"Found {len(tasks)} caps with PDFs on page {current_page_num}")
                results = await asyncio.gather(*tasks)
                success_count = sum(1 for r in results if r)
                print(f"Page {current_page_num} processed: {success_count}/{len(tasks)} downloaded.")
            else:
                print(f"No caps with English PDFs found on page {current_page_num}.")

            # Check if there's a "Next Page" link to continue
            next_button = await page.query_selector("a.grid-page-next")
            if not next_button:
                print("No 'Next Page' link found. Ending.")
                break
            
            current_page_num += 1

        if failed_caps:
            print("\nThe following caps failed to download:")
            for cap in failed_caps:
                print(f"  - {cap}")
        else:
            print("\nAll caps downloaded successfully (or already existed).")
        await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download legal PDFs.")
    parser.add_argument("--start", type=int, default=1, help="Start page number")
    parser.add_argument("--end", type=int, help="End page number")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent downloads")
    args = parser.parse_args()
    
    asyncio.run(batch_download(start_page=args.start, end_page=args.end, concurrency=args.concurrency))
