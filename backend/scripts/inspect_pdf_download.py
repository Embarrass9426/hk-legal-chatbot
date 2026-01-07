import asyncio
from playwright.async_api import async_playwright

async def inspect_download(cap_number):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Log all requests
        async def handle_response(response):
            # print(f"<< {response.status} {response.url} {response.headers.get('content-type')}")
            if "xml?skipHSC=true" in response.url:
                try:
                    text = await response.text()
                    with open("inspect_response.xml", "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Saved XML response to inspect_response.xml")
                except:
                    pass
            if "grid?skipHSC=true" in response.url:
                try:
                    text = await response.text()
                    print(f"GRID RESPONSE: {text}")
                except:
                    pass

        page.on("request", lambda request: print(f">> {request.method} {request.url}"))
        page.on("response", handle_response)

        print(f"Navigating to Cap {cap_number} page...")
        # Try to go to the main page for the cap first
        url = f"https://www.elegislation.gov.hk/hk/cap{cap_number}"
        await page.goto(url, wait_until="load", timeout=60000)
        content = await page.content()
        with open("inspect_page.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("Saved page HTML to inspect_page.html")
        
        print("Looking for PDF download link...")
        # The PDF download button usually has some specific text or icon
        # Let's try to find it.
        try:
            # Wait for the page to load content
            await page.wait_for_selector("a", timeout=10000)
            
            # Find links that might be PDF downloads
            pdf_links = await page.query_selector_all("a")
            for link in pdf_links:
                text = await link.inner_text()
                href = await link.get_attribute("href")
                if "PDF" in text.upper() or (href and ".pdf" in href.lower()):
                    print(f"Found potential PDF link: Text='{text}', Href='{href}'")
            
            # If we find a PDF button, click it and see what happens
            pdf_button = await page.query_selector("a:has-text('PDF')")
            if pdf_button:
                print("Clicking PDF button...")
                async with page.expect_download() as download_info:
                    await pdf_button.click()
                download = await download_info.value
                print(f"Download started: {download.url}")
                await download.save_as(f"inspect_cap_{cap_number}.pdf")
                print(f"Saved to inspect_cap_{cap_number}.pdf")
            else:
                print("PDF button not found.")
                
        except Exception as e:
            print(f"Error during inspection: {e}")
        
        await browser.close()

if __name__ == "__main__":
    import sys
    cap = sys.argv[1] if len(sys.argv) > 1 else "1"
    asyncio.run(inspect_download(cap))
