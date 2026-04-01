# Implementation Plan — HK Legal Chatbot Enhancement

## Overview

Five workstreams transforming the chatbot from a Cap. 282 specialist into a general HK legal assistant with smart retrieval, context summarization, and improved evaluation.

**Files modified:**
- `backend/main.py` (WS1, WS3, WS4, WS5)
- `backend/llm_evaluate.py` (WS1, WS2)

---

## WS1: Generalize System Prompts

### Goal
Change both `main.py` and `llm_evaluate.py` system prompts from Cap. 282 (Employees' Compensation) specialist to general HK legal assistant.

### Changes

**File: `backend/main.py` lines 563-572**
Replace the Cap. 282 specialized prompt:
```python
# CURRENT (line 563-572):
system_content = """You are an expert Hong Kong legal assistant specializing in the Employees' Compensation Ordinance (Cap. 282).
Your goal is to help employees understand their rights regarding workplace injuries and insurance.

Instructions:
1. Use the provided legal context to answer the user's question.
2. If the context doesn't contain the answer, state that you don't have enough information but provide general guidance based on the context.
3. Always cite the specific Section (e.g., [1] Cap. 282, s. 5) when referring to the law.
4. Be empathetic but professional.
5. If the user asks about a specific injury (like breaking a leg), explain if it's covered under "arising out of and in the course of employment".
"""

# NEW:
system_content = """You are an expert Hong Kong legal assistant with comprehensive knowledge of Hong Kong ordinances and regulations.
Your goal is to help users understand their legal rights and obligations under Hong Kong law.

Instructions:
1. Use the provided legal context to answer the user's question.
2. If the context doesn't contain the answer, state that you don't have enough information but provide general guidance based on the context.
3. Always cite the specific Section and Ordinance (e.g., [1] Cap. 282, s. 5) when referring to the law.
4. Be empathetic but professional.
5. Explain legal concepts in plain language while maintaining legal accuracy.
"""
```

**File: `backend/llm_evaluate.py` lines 299-303**
Replace the evaluation system prompt:
```python
# CURRENT (line 299-303):
system_prompt = (
    "You are an expert Hong Kong legal assistant. "
    "Answer the user's question based ONLY on the provided legal context. "
    "If the context doesn't contain enough information, say so. "
    "Always cite specific sections when possible."
)

# NEW:
system_prompt = (
    "You are an expert Hong Kong legal assistant with comprehensive knowledge of Hong Kong ordinances and regulations. "
    "Answer the user's question based ONLY on the provided legal context. "
    "If the context doesn't contain enough information, say so. "
    "Always cite the specific Section and Ordinance when possible."
)
```

### Verification
- Read back both files after edit to confirm changes
- Run `lsp_diagnostics` on both files

### QA Scenario (WS1)

**Command (verify main.py prompt):**
```powershell
python -c "import ast; tree=ast.parse(open('backend/main.py',encoding='utf-8').read()); src=open('backend/main.py',encoding='utf-8').read(); assert 'comprehensive knowledge of Hong Kong ordinances and regulations' in src, 'main.py prompt not updated'; assert 'specializing in the Employees' not in src, 'main.py still has old Cap. 282 prompt'; print('WS1 main.py OK')"
```

**Command (verify llm_evaluate.py prompt):**
```powershell
python -c "src=open('backend/llm_evaluate.py',encoding='utf-8').read(); assert 'comprehensive knowledge of Hong Kong ordinances and regulations' in src, 'eval prompt not updated'; assert 'Always cite the specific Section and Ordinance when possible' in src, 'eval citation instruction not updated'; print('WS1 llm_evaluate.py OK')"
```

**Expected:** Both print "OK". No assertion errors.

---

## WS2: Evaluation Script Improvements

### Goal
Three changes:
1. Grade `retrieval_relevance` using the **rewritten query** instead of the original user query
2. Add a **no_rag_baseline** strategy that generates answers without vector DB context
3. For no_rag_baseline: evaluate **helpfulness** (response relevance) and **groundedness vs retrieved docs** (compare the no-RAG answer to what the best RAG strategy retrieved, to measure hallucination)

### Change 2a: Pass rewritten_query to judge_retrieval_relevance

**File: `backend/llm_evaluate.py`**

The `evaluate_strategy` function (line 411) currently receives `query` (original) and passes it to all three judges. We need to also accept `rewritten_query` and pass it specifically to `judge_retrieval_relevance`.

```python
# CURRENT signature (line 411-417):
async def evaluate_strategy(
    strategy_name: str,
    query: str,
    context_text: str,
    answer: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:

# NEW signature:
async def evaluate_strategy(
    strategy_name: str,
    query: str,
    context_text: str,
    answer: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
    rewritten_query: str = "",
) -> Dict[str, Any]:
```

Inside the function (line 423-426), change `judge_retrieval_relevance` to use the rewritten query:
```python
# CURRENT (line 426):
_limited_call(judge_retrieval_relevance(query, context_text, llm)),

# NEW:
_limited_call(judge_retrieval_relevance(rewritten_query or query, context_text, llm)),
```

Update the call site in `run_evaluation` (line 1003-1010) to pass `rewritten_query`:
```python
# CURRENT (line 1003-1010):
evaluate_strategy(
    strategy,
    query,
    context_text,
    answer,
    llm,
    semaphore,
),

# NEW:
evaluate_strategy(
    strategy,
    query,
    context_text,
    answer,
    llm,
    semaphore,
    rewritten_query=rewritten_query,
),
```

### Change 2b: Add no_rag_baseline strategy

**File: `backend/llm_evaluate.py`**

Add a new function to generate answers without RAG context:
```python
async def generate_answer_no_rag(query: str, llm: ChatOpenAI) -> str:
    """Generate answer without any RAG context or system prompt — pure LLM."""
    messages = [
        HumanMessage(content=f"Please provide legal advice for the following question about Hong Kong law:\n\n{query}"),
    ]
    response = await llm.ainvoke(messages)
    return str(response.content).strip()
```

Add new judge function — `judge_groundedness_vs_docs`:
```python
async def judge_groundedness_vs_docs(
    answer: str, reference_context: str, llm: ChatOpenAI
) -> Dict[str, Any]:
    """Compare a no-RAG answer against retrieved docs to measure hallucination.
    
    reference_context is the context from the best RAG strategy (rewritten_expanded).
    We check how much of the no-RAG answer is supported by actual legal documents.
    """
    system_prompt = """You are an impartial evaluator.
Task: Evaluate how well the assistant's response (generated WITHOUT access to legal documents) is grounded in the actual legal documents provided as reference.

This measures hallucination — how much of the response is factually supported by real legal text.

Scoring Criteria (0.0–10.0):
- 0–2: Response contains mostly fabricated or incorrect legal information
- 3–4: Major legal claims are unsupported or incorrect
- 5–6: Some claims align with documents, but significant unsupported assertions
- 7–8: Most claims are supported, minor inaccuracies
- 9–10: All claims are verifiable against the reference documents

Evaluation Guidelines:
- Compare each legal claim in the response against the reference documents
- Penalize fabricated section numbers, case references, or legal principles
- Penalize incorrect legal conclusions not supported by the documents
- Reward accurate general principles even if specific citations are missing
"""
    user_prompt = (
        f"Reference Legal Documents (ground truth):\n{reference_context}\n\n"
        f"Assistant Response (generated WITHOUT documents):\n{answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}
```

Add new judge function — `judge_helpfulness`:
```python
async def judge_helpfulness(
    query: str, answer: str, llm: ChatOpenAI
) -> Dict[str, Any]:
    """Evaluate how helpful the response is to a user seeking legal advice."""
    system_prompt = """You are an impartial evaluator.
Task: Evaluate how helpful and actionable the assistant's legal advice is for the user.

Scoring Criteria (0.0–10.0):
- 0–2: Completely unhelpful, no useful information
- 3–4: Minimally helpful, vague or generic advice
- 5–6: Somewhat helpful, provides some guidance but lacks specificity
- 7–8: Helpful, provides actionable advice with reasonable specificity
- 9–10: Highly helpful, comprehensive, actionable, and well-structured advice

Evaluation Guidelines:
- Focus on practical usefulness to someone seeking legal guidance
- Reward specific, actionable steps or explanations
- Reward proper legal context and caveats (e.g., "consult a lawyer")
- Penalize vague, generic, or misleading advice
- Consider whether the advice addresses the user's actual situation
"""
    user_prompt = (
        f"User Query: {query}\n\n"
        f"Assistant Response: {answer}\n\n"
        'Respond in JSON: {"score": <0.0-10.0>, "reasoning": "..."}'
    )
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return safe_judge_parse(str(response.content))
    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge error: {exc}"}
```

Add a dedicated evaluation function for no-RAG baseline:
```python
async def evaluate_no_rag_baseline(
    query: str,
    no_rag_answer: str,
    reference_context: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Evaluate the no-RAG baseline answer.
    
    Metrics:
    - relevance: Does the answer address the query? (same judge as RAG strategies)
    - helpfulness: Is the advice actionable and useful?
    - groundedness_vs_docs: How much is supported by actual legal documents? (hallucination measure)
    """
    async def _limited_call(coro):
        async with semaphore:
            return await coro

    relevance, helpfulness, groundedness_vs_docs = await asyncio.gather(
        _limited_call(judge_relevance(query, no_rag_answer, llm)),
        _limited_call(judge_helpfulness(query, no_rag_answer, llm)),
        _limited_call(judge_groundedness_vs_docs(no_rag_answer, reference_context, llm)),
    )
    avg_score = (
        relevance["score"] + helpfulness["score"] + groundedness_vs_docs["score"]
    ) / 3
    return {
        "relevance": relevance,
        "helpfulness": helpfulness,
        "groundedness_vs_docs": groundedness_vs_docs,
        "avg_score": round(avg_score, 1),
    }
```

### Change 2c: Integrate no_rag_baseline into run_evaluation

In `run_evaluation`, add `"no_rag_baseline"` to strategies list (line 939):
```python
strategies = [
    "plain_vector",
    "rewritten_vector",
    "rewritten_expanded",
    "no_rag_baseline",
]
```

In the per-strategy loop (line 975-1032), add handling for no_rag_baseline:
```python
if strategy == "no_rag_baseline":
    # Generate answer without context
    no_rag_answer = await _run_with_semaphore(
        semaphore,
        generate_answer_no_rag(query, llm),
    )
    
    # Get reference context from rewritten_expanded for groundedness comparison
    # (we need to retrieve this — reuse the rewritten_expanded retrieval if done, 
    #  or do it here)
    ref_strategy_data = query_result["strategies"].get("rewritten_expanded", {})
    reference_context = ""
    if ref_strategy_data:
        # Build context from already-retrieved sections
        ref_sections = ref_strategy_data.get("retrieved_sections", [])
        reference_context = build_context_text_from_sections(ref_sections)
    
    scores = await evaluate_no_rag_baseline(
        query,
        no_rag_answer,
        reference_context,
        llm,
        semaphore,
    )
    
    strategy_result = {
        "answer": no_rag_answer,
        "num_sources": 0,
        "retrieved_sections": [],
        "source_usefulness": [],
        "scores": scores,
    }
    query_result["strategies"][strategy] = strategy_result
    progress.update(1)
    continue  # Skip the rest of the loop body
```

**Important**: `no_rag_baseline` must run AFTER `rewritten_expanded` so we have reference context. The strategies list order already ensures this.

### Change 2d: Update compute_strategy_summary

In `compute_strategy_summary` (line 455-494), update the hardcoded strategy list:
```python
# CURRENT (line 458-462):
strategy_names = [
    "plain_vector",
    "rewritten_vector",
    "rewritten_expanded",
]

# NEW — dynamically collect from data:
strategy_names = set()
for item in details:
    strategy_names.update(item.get("strategies", {}).keys())
strategy_names = sorted(strategy_names)
```

The score aggregation loop already uses `.get()` with defaults, so it handles missing keys gracefully. For no_rag_baseline, it will have `helpfulness` and `groundedness_vs_docs` instead of `groundedness` and `retrieval_relevance`. We need to handle this:

```python
for strategy in strategy_names:
    relevance_scores = []
    groundedness_scores = []
    retrieval_relevance_scores = []
    helpfulness_scores = []
    groundedness_vs_docs_scores = []
    overall_scores = []

    for item in details:
        strategy_data = item.get("strategies", {}).get(strategy, {})
        scores = strategy_data.get("scores", {})
        relevance_scores.append(float(scores.get("relevance", {}).get("score", 0)))
        groundedness_scores.append(float(scores.get("groundedness", {}).get("score", 0)))
        retrieval_relevance_scores.append(float(scores.get("retrieval_relevance", {}).get("score", 0)))
        helpfulness_scores.append(float(scores.get("helpfulness", {}).get("score", 0)))
        groundedness_vs_docs_scores.append(float(scores.get("groundedness_vs_docs", {}).get("score", 0)))
        overall_scores.append(float(scores.get("avg_score", 0)))

    count = len(details) if details else 1
    strat_summary = {
        "avg_relevance": round(sum(relevance_scores) / count, 1),
        "avg_groundedness": round(sum(groundedness_scores) / count, 1),
        "avg_retrieval_relevance": round(sum(retrieval_relevance_scores) / count, 1),
        "avg_overall": round(sum(overall_scores) / count, 1),
    }
    
    # Add no-RAG-specific metrics only if present
    if any(s > 0 for s in helpfulness_scores):
        strat_summary["avg_helpfulness"] = round(sum(helpfulness_scores) / count, 1)
    if any(s > 0 for s in groundedness_vs_docs_scores):
        strat_summary["avg_groundedness_vs_docs"] = round(sum(groundedness_vs_docs_scores) / count, 1)
    
    summary[strategy] = strat_summary
```

### Change 2e: Update _collect_strategy_score_row and print summary

Update `_collect_strategy_score_row` (line 660-666) to include new metrics:
```python
def _collect_strategy_score_row(scores: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "relevance": scores.get("relevance", {}).get("score", 0),
        "groundedness": scores.get("groundedness", {}).get("score", 0),
        "retrieval_relevance": scores.get("retrieval_relevance", {}).get("score", 0),
        "avg_score": scores.get("avg_score", 0),
    }
    # Include no-RAG-specific metrics if present
    if "helpfulness" in scores:
        row["helpfulness"] = scores["helpfulness"].get("score", 0)
    if "groundedness_vs_docs" in scores:
        row["groundedness_vs_docs"] = scores["groundedness_vs_docs"].get("score", 0)
    return row
```

Update the print summary (line 1116-1130) to handle variable column layout:
```python
# Replace the hardcoded table header and row format:
if not progress_only:
    print("\n[Eval] Summary")
    # Determine if any strategy has no-RAG-specific metrics
    has_helpfulness = any(
        "avg_helpfulness" in scores
        for scores in summary["by_strategy"].values()
    )
    
    header = (
        "Strategy                         "
        "| Relevance | Groundedness | RetrievalRel"
    )
    if has_helpfulness:
        header += " | Helpfulness | Ground.Docs"
    header += " | Overall"
    print(header)
    print("-" * len(header))
    
    for strategy_name, scores in summary["by_strategy"].items():
        row = (
            f"{strategy_name:<32} | "
            f"{scores.get('avg_relevance', 0):>9.1f} | "
            f"{scores.get('avg_groundedness', 0):>11.1f} | "
            f"{scores.get('avg_retrieval_relevance', 0):>12.1f}"
        )
        if has_helpfulness:
            row += (
                f" | {scores.get('avg_helpfulness', 0):>11.1f}"
                f" | {scores.get('avg_groundedness_vs_docs', 0):>11.1f}"
            )
        row += f" | {scores.get('avg_overall', 0):>7.1f}"
        print(row)
```

### Verification
- Run `lsp_diagnostics` on `backend/llm_evaluate.py`
- Confirm no syntax errors

### QA Scenario (WS2)

**Command:**
```powershell
python backend\llm_evaluate.py
```
(Runs full evaluation pipeline against `backend/data/queries.jsonl` with all strategies.)

**Expected behavior:**
1. Console progress bar shows 4 strategies per query (not 3): `plain_vector`, `rewritten_vector`, `rewritten_expanded`, `no_rag_baseline`
2. For `no_rag_baseline`, the log should NOT show any vector search calls — only LLM generation
3. The summary table printed at the end includes a `no_rag_baseline` row with columns for relevance, helpfulness, and groundedness_vs_docs
4. `eval_scores.json` at `$.general_summary.by_strategy` contains key `"no_rag_baseline"` with fields:
   - `avg_relevance` (number)
   - `avg_helpfulness` (number)
   - `avg_groundedness_vs_docs` (number)
   - `avg_overall` (number)
5. For the 3 RAG strategies, `eval_scores.json` should show `retrieval_relevance` scores **different** from the previous run (since we now grade with `rewritten_query` instead of original `query`)
6. `eval_results.json` for each query's `no_rag_baseline` entry has `"num_sources": 0` and `"retrieved_sections": []`

**Validation command (quick sanity check on output file):**
```powershell
python -c "import json; d=json.load(open('backend/data/eval_scores.json')); s=d['general_summary']['by_strategy']; assert 'no_rag_baseline' in s, 'missing no_rag_baseline'; assert 'avg_helpfulness' in s['no_rag_baseline'], 'missing helpfulness metric'; print('WS2 OK:', json.dumps(s['no_rag_baseline'], indent=2))"
```

---

## WS3: Smart Retrieval via Tool-Use / Function Calling

### Goal
Let the LLM decide when to search the legal database instead of always searching. Use OpenAI-compatible function calling (DeepSeek supports this).

### Design

Define a `search_legal_database` tool. The flow becomes:

1. User sends message
2. Build messages with system prompt + chat history + user message
3. Call LLM with `tools=[search_legal_database]` (non-streaming first call)
4. If LLM calls the tool → execute search → build context → call LLM again with context (streaming)
5. If LLM doesn't call the tool → stream the direct response

### Changes

**File: `backend/main.py`**

Add tool definition (new constant near top of file, after line 35):
```python
SEARCH_LEGAL_DB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_legal_database",
        "description": (
            "Search the Hong Kong legal database for relevant ordinances, regulations, "
            "and legal provisions. Use this tool when the user asks a question that requires "
            "legal references, citations, or specific legal information. "
            "Do NOT use this tool for greetings, casual conversation, clarification questions, "
            "or follow-up questions that can be answered from the conversation history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query optimized for legal document retrieval. Rewrite the user's question into a clear, precise legal query.",
                }
            },
            "required": ["query"],
        },
    },
}
```

Rewrite `generate_chat_responses` (line 505-631):

```python
async def generate_chat_responses(
    message: str,
    language: str = "en",
    session_id: str = "default",
):
    normalized_session_id = _normalize_session_id(session_id)
    if normalized_session_id == "default":
        yield "data: " + json.dumps({"error": "session_id is required."}) + "\n\n"
        return

    memory = await _get_or_create_conversation_memory(normalized_session_id)
    async with memory.stream_lock:
        try:
            # Step 1: Build base system prompt (no context yet)
            base_system = _build_general_system_prompt(language)
            
            # Step 2: Add user turn to memory
            async with memory.lock:
                _append_turn(memory, "user", message)
                await _compact_memory_if_needed(memory, get_llm(), base_system)
                memory_messages = _build_memory_messages(memory)

            # Step 3: First LLM call WITH tools — let it decide whether to search
            messages_for_tool_decision = [
                SystemMessage(content=base_system),
                *memory_messages,
            ]
            
            tool_response = await get_llm().ainvoke(
                messages_for_tool_decision,
                tools=[SEARCH_LEGAL_DB_TOOL],
                tool_choice="auto",
            )

            context_text = ""
            sections = []
            references = []

            # Step 4: Check if tool was called
            if tool_response.tool_calls:
                tool_call = tool_response.tool_calls[0]
                search_query = tool_call["args"].get("query", message)
                
                # Execute the search
                retrieval_query = await rewrite_query(message, get_llm())
                search_results = vector_manager.search(retrieval_query, k=5)
                sections = collapse_documents_to_sections(search_results)
                
                # Build context and references
                context_parts, references = _build_context_and_references(sections)
                context_text = "\n\n".join(context_parts)
                
                # Check if context exceeds 42k tokens — if so, map-reduce summarize
                context_tokens = _estimate_text_tokens(context_text)
                if context_tokens > DOCS_CONTEXT_BUDGET_TOKENS:
                    context_text = await _map_reduce_summarize(
                        context_parts, message, get_llm()
                    )
                
                # Rebuild system prompt with context
                system_with_context = base_system + "\n\nCONTEXT:\n" + context_text
                
                # Step 5: Second LLM call — streaming with context
                async with memory.lock:
                    memory_messages = _build_memory_messages(memory)
                
                messages = [SystemMessage(content=system_with_context), *memory_messages]
            else:
                # No tool call — LLM will answer directly
                # Stream the response from the tool decision response
                # Actually, we need to stream — so we do a second call
                messages = [SystemMessage(content=base_system), *memory_messages]
            
            # Step 6: Stream the final response
            answer_parts = []
            async for chunk in get_llm().astream(messages):
                chunk_text = _safe_str(chunk.content, "")
                if chunk_text:
                    answer_parts.append(chunk_text)
                    yield f"data: {json.dumps({'answer': chunk_text})}\n\n"

            full_answer = "".join(answer_parts).strip()
            if full_answer:
                async with memory.lock:
                    _append_turn(memory, "assistant", full_answer)
                    await _compact_memory_if_needed(memory, get_llm(), base_system)

            # Source usefulness assessment (only if we searched)
            if sections:
                source_usefulness = await assess_section_usefulness(
                    message, full_answer, sections
                )
                references_with_usefulness = []
                for idx, reference in enumerate(references, start=1):
                    usefulness = (
                        source_usefulness[idx - 1]
                        if idx - 1 < len(source_usefulness)
                        else {"source_index": idx, "is_useful": False, "usefulness_score": 0.0, "reasoning": "Missing."}
                    )
                    references_with_usefulness.append({**reference, **usefulness})
                
                if references_with_usefulness:
                    yield f"data: {json.dumps({'references': references_with_usefulness})}\n\n"

        except Exception as e:
            print(f"Error in stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
```

Add helper functions:
```python
def _build_general_system_prompt(language: str) -> str:
    """Build the generalized system prompt."""
    prompt = """You are an expert Hong Kong legal assistant with comprehensive knowledge of Hong Kong ordinances and regulations.
Your goal is to help users understand their legal rights and obligations under Hong Kong law.

Instructions:
1. Use the provided legal context to answer the user's question.
2. If the context doesn't contain the answer, state that you don't have enough information but provide general guidance based on the context.
3. Always cite the specific Section and Ordinance (e.g., [1] Cap. 282, s. 5) when referring to the law.
4. Be empathetic but professional.
5. Explain legal concepts in plain language while maintaining legal accuracy.
"""
    if language == "zh":
        prompt += "\nIMPORTANT: You MUST respond in Traditional Chinese (繁體中文). Even if the context is in English, translate the relevant parts to help the user understand, but keep the legal citations (e.g. Cap. 282, s. 5) recognizable."
    else:
        prompt += "\nIMPORTANT: Respond in English."
    return prompt


def _build_context_and_references(
    sections: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build context parts and references from sections."""
    context_parts = []
    references = []
    for i, section in enumerate(sections):
        pages = section.get("pages", [])
        pages_text = ", ".join(pages) if pages else "Unknown"
        context_parts.append(
            f"### Reference [{i + 1}]: {section.get('section_title', 'Unknown Section')}\n"
            f"Citation: {section.get('citation', 'Unknown')}\n"
            f"Pages: {pages_text}\n"
            f"{section.get('content', '')}"
        )
        references.append({
            "id": f"ref-{i}",
            "doc_id": section.get("doc_id", "Unknown"),
            "section_id": section.get("section_id", "Unknown"),
            "section_title": section.get("section_title", "Unknown Section"),
            "title": section.get("citation", "Unknown"),
            "source_url": section.get("source_url", ""),
            "pages": pages,
            "jurisdiction": "Hong Kong",
            "type": "Ordinance",
        })
    return context_parts, references
```

### DeepSeek Function Calling Compatibility Note
DeepSeek API supports OpenAI-compatible function calling. The `ChatOpenAI` from `langchain_openai` passes `tools` and `tool_choice` to the API. Response will have `tool_calls` attribute on the AIMessage.

### Verification
- `lsp_diagnostics` on `backend/main.py`
- Manual test: send "hello" → should NOT trigger search
- Manual test: send "what are my rights under Cap 282?" → SHOULD trigger search

### QA Scenario (WS3)

**Prerequisites:** Start the backend server:
```powershell
python backend\main.py
```
Server starts at `http://localhost:8000`. Wait for `Uvicorn running on http://0.0.0.0:8000` message.

**Test 1 — Greeting (no search expected):**
```powershell
$body = '{"message": "Hello, how are you?", "session_id": "test-ws3-greeting"}'
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body $body
```

**Expected:**
- Response is SSE stream: `data: {"answer": "..."}` chunks containing a friendly greeting
- NO `data: {"references": [...]}` event at end (no search was triggered)
- Response should NOT contain legal citations like `[1] Cap. 282, s. X`

**Test 2 — Legal question (search expected):**
```powershell
$body = '{"message": "What are my rights if I get injured at work in Hong Kong?", "session_id": "test-ws3-legal"}'
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body $body
```

**Expected:**
- Response is SSE stream with `data: {"answer": "..."}` chunks
- Answer contains legal citations (e.g., `[1] Cap. 282, s. 5` or similar)
- Final SSE event includes `data: {"references": [...]}` with at least 1 reference object containing `doc_id`, `section_title`, `title`, `pages`
- Each reference has a `usefulness_score` and `is_useful` field

**Test 3 — Follow-up (no search expected for conversational follow-up):**
```powershell
$body = '{"message": "Can you explain that in simpler terms?", "session_id": "test-ws3-legal"}'
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body $body
```

**Expected:**
- LLM uses conversation history (same session_id) to answer
- NO new `data: {"references": [...]}` event (no new search triggered)
- Response rephrases previous answer in simpler language

---

## WS4: Map-Reduce Summarization for Oversized Context

### Goal
When retrieved legal documents exceed 42,000 tokens, apply map-reduce summarization to compress the context while preserving key legal information.

### Design
- **Threshold**: 42,000 tokens (DOCS_CONTEXT_BUDGET_TOKENS)
- **Map step**: Summarize each section independently (parallel)
- **Reduce step**: Combine all summaries into a final condensed context
- Uses the user's prompt templates (provided earlier in conversation)

### Changes

**File: `backend/main.py`**

Add constants (near line 32):
```python
DOCS_CONTEXT_BUDGET_TOKENS = 42_000
HISTORY_CONTEXT_WINDOW_TOKENS = 42_000  # Changed from 128_000
SUMMARY_TRIGGER_RATIO = 0.7
SUMMARY_TRIGGER_TOKENS = int(HISTORY_CONTEXT_WINDOW_TOKENS * SUMMARY_TRIGGER_RATIO)  # ~29,400
```

Add map-reduce summarization function:
```python
async def _map_reduce_summarize(
    context_parts: List[str],
    user_query: str,
    llm_client: ChatOpenAI,
) -> str:
    """Map-reduce summarize context when it exceeds the token budget.
    
    Map step: summarize each section independently (parallel).
    Reduce step: combine all summaries into a single condensed context.
    """
    # Map step — summarize each section
    map_prompt = """You are a legal document summarizer. 
Extract the key legal provisions, rules, rights, obligations, and any specific conditions or exceptions from the following legal text.
Focus on information that would be relevant to answering legal questions.
Preserve all section numbers, citations, and specific legal terms exactly.
Be concise but do not omit legally significant details.

Legal text to summarize:
{text}

User's question (for context on what's relevant): {query}
"""
    
    async def _summarize_section(section_text: str) -> str:
        messages = [
            HumanMessage(content=map_prompt.format(text=section_text, query=user_query))
        ]
        response = await llm_client.ainvoke(messages)
        return _safe_str(response.content, section_text).strip()

    # Run map step in parallel
    map_tasks = [_summarize_section(part) for part in context_parts]
    summaries = await asyncio.gather(*map_tasks)

    # Check if summaries fit in budget
    combined = "\n\n".join(summaries)
    if _estimate_text_tokens(combined) <= DOCS_CONTEXT_BUDGET_TOKENS:
        return combined

    # Reduce step — combine summaries into final context
    reduce_prompt = f"""You are a legal document summarizer performing a final consolidation.
Below are summaries of multiple legal document sections relevant to a user's question.
Combine them into a single, coherent legal reference that:
1. Preserves all unique legal provisions, rules, and conditions
2. Removes redundant information across sections
3. Maintains all section numbers and citations exactly
4. Stays within a concise format suitable for answering the user's question

User's question: {user_query}

Section summaries:
{combined}

Provide the consolidated legal reference:"""

    messages = [HumanMessage(content=reduce_prompt)]
    response = await llm_client.ainvoke(messages)
    return _safe_str(response.content, combined).strip()
```

### Integration Point
This is called inside `generate_chat_responses` (WS3) after building context_parts:
```python
context_tokens = _estimate_text_tokens(context_text)
if context_tokens > DOCS_CONTEXT_BUDGET_TOKENS:
    context_text = await _map_reduce_summarize(context_parts, message, get_llm())
```

### Verification
- `lsp_diagnostics` on `backend/main.py`
- Token estimation test: 5 sections * 1200 tokens = 6000 tokens (well under 42k, so map-reduce should rarely trigger in practice — as user noted)

### QA Scenario (WS4 — Map-Reduce + WS5 — Constants)

**Step 1 — Verify constants are correct:**
```powershell
python -c "import sys; sys.path.insert(0,'.'); from backend.main import HISTORY_CONTEXT_WINDOW_TOKENS, DOCS_CONTEXT_BUDGET_TOKENS, SUMMARY_TRIGGER_TOKENS, RECENT_TURNS_TO_KEEP; assert HISTORY_CONTEXT_WINDOW_TOKENS == 42_000, f'Expected 42000, got {HISTORY_CONTEXT_WINDOW_TOKENS}'; assert DOCS_CONTEXT_BUDGET_TOKENS == 42_000, f'Expected 42000, got {DOCS_CONTEXT_BUDGET_TOKENS}'; assert SUMMARY_TRIGGER_TOKENS == 29_400, f'Expected 29400, got {SUMMARY_TRIGGER_TOKENS}'; assert RECENT_TURNS_TO_KEEP == 12; print('WS5 constants OK')"
```

**Step 2 — Unit test for map-reduce function:**

Create and run a temporary test script that exercises `_map_reduce_summarize` directly:
```powershell
python -c "
import asyncio, sys, os
sys.path.insert(0, '.')
os.environ.setdefault('DEEPSEEK_API_KEY', os.getenv('DEEPSEEK_API_KEY', ''))
from backend.main import _map_reduce_summarize, _estimate_text_tokens, DOCS_CONTEXT_BUDGET_TOKENS, get_llm

async def test():
    # Create synthetic context that exceeds 42k tokens (~168k chars at 4 chars/token)
    parts = []
    for i in range(50):
        # Each part ~3400 chars = ~850 tokens. 50 parts = ~42,500 tokens (just over budget)
        parts.append(f'### Section {i+1}: Test Ordinance\nCitation: Cap. {100+i}, s. {i+1}\n' + ('This is a test legal provision about workplace safety requirements. ' * 80))

    total_tokens = _estimate_text_tokens('\n\n'.join(parts))
    print(f'Input tokens: {total_tokens} (budget: {DOCS_CONTEXT_BUDGET_TOKENS})')
    assert total_tokens > DOCS_CONTEXT_BUDGET_TOKENS, f'Test setup error: input ({total_tokens}) must exceed budget ({DOCS_CONTEXT_BUDGET_TOKENS})'

    result = await _map_reduce_summarize(parts, 'What are workplace safety requirements?', get_llm())
    result_tokens = _estimate_text_tokens(result)
    print(f'Output tokens: {result_tokens}')
    assert result_tokens < total_tokens, f'Summarization must reduce tokens: {result_tokens} >= {total_tokens}'
    assert len(result) > 100, f'Result too short ({len(result)} chars), possible error'
    print('WS4 map-reduce OK')

asyncio.run(test())
"
```

**Expected:**
- Input tokens printed > 42,000
- `_map_reduce_summarize` runs without error
- Output tokens < input tokens (summarization reduced context)
- Result contains meaningful text (> 100 chars)
- Both print statements show "OK"

**Note:** This test makes real DeepSeek API calls. Requires `DEEPSEEK_API_KEY` in `.env`.

---

## WS5: Update Context Window Allocation

### Goal
Set context window budget: 42k tokens for chat history, 42k tokens for docs.

### Changes

**File: `backend/main.py` lines 32-35**
```python
# CURRENT:
HISTORY_CONTEXT_WINDOW_TOKENS = 128_000
SUMMARY_TRIGGER_RATIO = 0.7
SUMMARY_TRIGGER_TOKENS = int(HISTORY_CONTEXT_WINDOW_TOKENS * SUMMARY_TRIGGER_RATIO)
RECENT_TURNS_TO_KEEP = 12

# NEW:
HISTORY_CONTEXT_WINDOW_TOKENS = 42_000
DOCS_CONTEXT_BUDGET_TOKENS = 42_000
SUMMARY_TRIGGER_RATIO = 0.7
SUMMARY_TRIGGER_TOKENS = int(HISTORY_CONTEXT_WINDOW_TOKENS * SUMMARY_TRIGGER_RATIO)  # ~29,400
RECENT_TURNS_TO_KEEP = 12
```

### Verification
- Confirm constants are correct (covered by WS4/WS5 QA scenario Step 1 above)
- `lsp_diagnostics`

---

## Implementation Order

1. **WS1** (system prompts) — simplest, no dependencies
2. **WS5** (constants) — just number changes
3. **WS4** (map-reduce) — new function, no restructuring
4. **WS3** (tool-use) — largest change, restructures generate_chat_responses
5. **WS2** (eval changes) — independent file, can be done in parallel with WS3/4

## Risk Notes

- **DeepSeek function calling**: Verify DeepSeek API supports `tools` parameter. If not, fall back to prompt-based routing (ask LLM "should I search?" in a structured prompt).
- **Streaming after tool call**: The second LLM call after tool execution streams normally. The first call (tool decision) is non-streaming.
- **No-RAG baseline depends on rewritten_expanded**: Strategy order matters in eval loop. Ensure `no_rag_baseline` runs last.
