import asyncio
from playwright.async_api import async_playwright

async def save_index_html():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        url = "https://www.elegislation.gov.hk/index/chapternumber?TYPE=1&TYPE=2&TYPE=3&LANGUAGE=E&p0=1"
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(10)
        
        content = await page.content()
        with open("index_page_1.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("Saved index_page_1.html")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(save_index_html())
