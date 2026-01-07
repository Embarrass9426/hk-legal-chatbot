import asyncio
from playwright.async_api import async_playwright

async def inspect_index():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        url = "https://www.elegislation.gov.hk/index/chapternumber?TYPE=1&TYPE=2&TYPE=3&LANGUAGE=E&p0=1"
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(5) # Give it time for dynamic content
        
        await page.screenshot(path="index_page_1.png")
        print("Screenshot saved to index_page_1.png")
        
        # Also print some link info
        links = await page.query_selector_all("a")
        for link in links[:50]:
            text = await link.inner_text()
            href = await link.get_attribute("href")
            if href and ("pdf" in href.lower() or "cap" in href.lower()):
                print(f"Link: {text.strip()} -> {href}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(inspect_index())
