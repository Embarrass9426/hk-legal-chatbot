import httpx
import asyncio
import os

async def test_download():
    cap_numbers = ["1", "A1", "282"]
    for cap in cap_numbers:
        url = f"https://www.elegislation.gov.hk/hk/cap{cap}!en.pdf"
        print(f"Testing {cap}: {url}")
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=10.0)
                print(f"  Status: {response.status_code}")
                print(f"  Content-Type: {response.headers.get('Content-Type')}")
                if response.status_code == 200:
                    print(f"  Size: {len(response.content)} bytes")
                    with open(f"test_cap_{cap}.pdf", "wb") as f:
                        f.write(response.content)
                else:
                    print(f"  Failed")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_download())
