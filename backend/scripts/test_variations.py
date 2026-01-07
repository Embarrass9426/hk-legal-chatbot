import httpx
import asyncio

async def test_urls():
    cap = "1"
    urls = [
        f"https://www.elegislation.gov.hk/hk/cap{cap}!en.pdf",
        f"https://www.elegislation.gov.hk/hk/cap{cap}!en.pdf?FROMCAPINDEX=Y",
        f"https://www.elegislation.gov.hk/jsp/ert/erts0114.jsp?PUBLISHED=true&LEG_VERSION_ID=44211&DOWNLOAD_MODE=PDF&HAS_ENGLISH=true&HAS_CHINESE=true&IS_NO_LONGER_IN_EFFECT=false&CAP_NO=1&VERSION_STATUS=I&DOC_FORMAT=P",
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        for url in urls:
            print(f"Testing URL: {url}")
            try:
                response = await client.get(url, timeout=10.0)
                print(f"  Status: {response.status_code}")
                print(f"  Content-Type: {response.headers.get('Content-Type')}")
                print(f"  Final URL: {response.url}")
                if "application/pdf" in response.headers.get("Content-Type", ""):
                    print(f"  SUCCESS! Size: {len(response.content)} bytes")
                else:
                    print(f"  FAILED: Not a PDF")
                    # print(f"  Body snippet: {response.text[:200]}")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_urls())
