import asyncio
from playwright.async_api import async_playwright

async def find_pdf_link(cap_number):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Go to the search/index page or direct cap page if possible
        # The URL for a specific cap HTML view is usually:
        url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}"
        print(f"Navigating to {url}...")
        await page.goto(url)
        
        # Wait for the page to load
        await asyncio.sleep(10)
        
        # Take a screenshot to debug
        await page.screenshot(path="debug_cap_page.png")
        print("Screenshot saved to debug_cap_page.png")

        # Find the PDF link and click it
        pdf_link = page.get_by_text("PDF", exact=True).first
        if await pdf_link.count() > 0:
            print("Clicking PDF link...")
            await pdf_link.click()
            await asyncio.sleep(2)
            
            # Look for the English download link
            eng_link = page.get_by_text("Whole document (English)", exact=False).first
            if await eng_link.count() == 0:
                eng_link = page.get_by_text("English", exact=False).first
            
            if await eng_link.count() > 0:
                print("Found English link, attempting to capture download...")
                try:
                    async with page.expect_download(timeout=10000) as download_info:
                        await eng_link.click()
                    download = await download_info.value
                    print(f"Download started! URL: {download.url}")
                    # Save the download to a file
                    await download.save_as(f"cap_{cap_number}.pdf")
                    print(f"Saved to cap_{cap_number}.pdf")
                except Exception as e:
                    print(f"Failed to capture download: {e}")
                    await page.screenshot(path="debug_after_click_fail.png")
            else:
                print("Could not find English PDF link after clicking PDF.")
                await page.screenshot(path="debug_after_click.png")

        else:
            print("PDF link not found.")




        await browser.close()

if __name__ == "__main__":
    asyncio.run(find_pdf_link("282"))
