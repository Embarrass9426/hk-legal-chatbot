import httpx, json, sys

try:
    response = httpx.post(
        "http://localhost:8000/chat",
        json={
            "message": "What are the requirements for divorce in Hong Kong?",
            "language": "en",
            "session_id": "qa-test-123"
        },
        timeout=180.0,
        headers={"Accept": "text/event-stream"}
    )
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        lines = response.text.strip().split("\n")
        print(f"Total SSE lines: {len(lines)}")

        statuses = []
        answer_parts = []
        refs_found = False
        error_found = False

        for line in lines:
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    if "status" in payload:
                        statuses.append(payload["status"])
                    if "answer" in payload:
                        answer_parts.append(payload["answer"])
                    if "references" in payload:
                        refs = payload["references"]
                        print(f"\nReferences ({len(refs)}):")
                        for r in refs[:2]:
                            print(f"  - {r.get('citation', 'N/A')}: {r.get('title', '')[:80]}...")
                        refs_found = True
                    if "error" in payload:
                        print(f"ERROR: {payload['error']}")
                        error_found = True
                except json.JSONDecodeError:
                    pass

        print(f"\nStatuses received: {statuses}")
        full_answer = "".join(answer_parts)
        print(f"Answer length: {len(full_answer)} chars")
        print(f"Success: answer={bool(full_answer)}, refs={refs_found}, errors={error_found}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text[:500])
except Exception as e:
    print(f"Exception: {e}")
