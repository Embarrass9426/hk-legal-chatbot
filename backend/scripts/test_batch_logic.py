import asyncio
import os
from playwright.async_api import async_playwright

async def test_batch():
    caps = ["1", "2"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("Visiting home page...")
        await page.goto("https://www.elegislation.gov.hk/", wait_until="load")
        await asyncio.sleep(5)
        
        for cap in caps:
            url = f"https://www.elegislation.gov.hk/hk/cap{cap}!en.pdf?FROMCAPINDEX=Y"
            print(f"Downloading Cap {cap} from {url}...")
            try:
                async with page.expect_download(timeout=30000) as download_info:
                    try:
                        await page.goto(url, wait_until="commit")
                    except Exception as e:
                        if "Download is starting" in str(e):
                            print("  Confirmed: Download starting...")
                        else:
                            raise e
                download = await download_info.value
                await download.save_as(f"test_cap{cap}.pdf")
                print(f"  Saved test_cap{cap}.pdf")
            except Exception as e:
                print(f"  Failed to download Cap {cap}: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_batch())
