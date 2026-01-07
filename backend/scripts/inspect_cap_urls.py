import asyncio
from playwright.async_api import async_playwright

async def check_caps():
    caps = ["6D", "9", "9A", "1B"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for cap in caps:
            print(f"--- Checking Cap {cap} ---")
            # 1. Check direct English PDF
            url_en = f"https://www.elegislation.gov.hk/hk/cap{cap}!en.pdf"
            try:
                response = await page.goto(url_en, timeout=10000)
                print(f"EN URL: {response.status} {response.headers.get('content-type')}")
            except Exception as e:
                print(f"EN URL Error: {e}")

            # 2. Check Bilingual PDF
            url_bi = f"https://www.elegislation.gov.hk/hk/cap{cap}!en-zh-Hant-HK.pdf"
            try:
                response = await page.goto(url_bi, timeout=10000)
                print(f"BI URL: {response.status} {response.headers.get('content-type')}")
            except Exception as e:
                print(f"BI URL Error: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(check_caps())
