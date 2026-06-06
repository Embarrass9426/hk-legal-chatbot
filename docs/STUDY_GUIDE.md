# HK Legal Chatbot Study Guide

This document is a repo-wide study guide for relearning this project from scratch.

It has two goals:

1. Explain the whole system in plain words so a reader can understand the repo structure without opening every file first.
2. Explain the important mistakes, pivots, and failure modes that shaped the current architecture, especially around PDF ingestion, embeddings, TensorRT, ONNX Runtime, CUDA, worker/queue design, and the Windows/WSL split.

This guide is **source-first**.

- **Primary authority**: executable sources and config in `backend/**/*.py`, `frontend/src/**`, `backend/requirements.txt`, and `frontend/package.json`.
- **Secondary authority**: internal docs like `PROJECT_LOG.md`, `docs/RAG_DESIGN.md`, `docs/PDF_PARSER_LOGIC.md`, and `.sisyphus/*` planning notes.
- If a design doc and the code disagree, trust the code.

---

## 1. What this repo is

At the highest level, this repo is a **Hong Kong legal RAG chatbot** with two runtime halves:

- a **frontend** written in React + Vite that lets the user ask questions and receive streaming answers with references
- a **backend** written in FastAPI that performs retrieval, memory handling, and streaming response generation

The backend does not just call an LLM directly. It first tries to retrieve relevant legal text from a Pinecone vector index built from Hong Kong ordinance PDFs. That means the project has **two different major runtime stories**:

1. **Chat/runtime path**: user asks a question -> backend retrieves legal context -> backend streams answer and references back to the browser.
2. **Ingestion/build path**: PDFs are parsed -> chunked -> embedded -> uploaded to Pinecone so retrieval has something to search later.

Those two stories share one very important dependency: the **embedding runtime**. That is why so much of the repo's complexity ends up concentrated around `EmbeddingService`, `VectorStoreManager`, ingestion queueing, provider setup, TensorRT caching, and platform-specific runtime stability.

### Real entrypoints

The repo's top-level AGENTS guide is explicit about the real boundaries:

- backend entry: `backend/main.py` (`AGENTS.md:10-18`)
- frontend entry: `frontend/src/main.jsx` (`AGENTS.md:10-18`)
- retrieval core: `backend/services/vector_store.py` (`AGENTS.md:13-18`)
- Ollama runtime probing/failover: `backend/core/ollama_runtime.py` (`AGENTS.md:13-18`)
- frontend stream/session logic: `frontend/src/components/ChatInterface.jsx` (`AGENTS.md:13-18`)

One more important repo-level rule is also stated clearly: prefer `backend/core/*` and `backend/services/*` over root-level duplicates such as `backend/vector_store.py`, `backend/utils.py`, `backend/setup_env.py`, and `backend/embedding_shared.py` (`AGENTS.md:18`).

That matters because when you relearn this project, it is very easy to accidentally read a stale or duplicate file and form the wrong mental model.

---

## 2. How to read this guide

Read this guide in this order:

1. repo map
2. end-to-end walkthrough
3. frontend flow
4. backend chat flow
5. retrieval pipeline
6. ingestion pipeline
7. worker/queue architecture
8. runtime stack and postmortem

That order mirrors how the system is easiest to understand:

- first, what the user sees
- second, what the server does
- third, how the indexed knowledge base is produced
- finally, why the GPU/runtime engineering became tricky

When you see references like `backend/main.py:898-1052`, that means the claim is tied to that exact source range.

---

## 3. Repo map and package boundaries

Here is the practical map, not the full tree.

```text
hk-legal-chatbot/
├─ frontend/
│  ├─ src/
│  │  ├─ main.jsx
│  │  ├─ App.jsx
│  │  ├─ index.css
│  │  └─ components/
│  │     ├─ ChatInterface.jsx
│  │     └─ ReferenceCard.jsx
│  ├─ package.json
│  ├─ vite.config.js
│  └─ eslint.config.js
├─ backend/
│  ├─ main.py
│  ├─ core/
│  │  ├─ setup_env.py
│  │  ├─ utils.py
│  │  ├─ embedding_shared.py
│  │  └─ ollama_runtime.py
│  ├─ services/
│  │  ├─ vector_store.py
│  │  ├─ embedding_service.py
│  │  └─ reranker_service.py
│  ├─ parsers/
│  │  └─ pdf_parser.py
│  ├─ scripts/
│  │  ├─ ingest_pdfs.py
│  │  ├─ verify_cpu_rerank_pipeline.py
│  │  ├─ export_yuan_trt.py
│  │  ├─ export_yuan_cuda.py
│  │  ├─ export_reranker_onnx.py
│  │  ├─ export_zerank_onnx.py
│  │  ├─ ollama_discover.py
│  │  ├─ ollama_lifecycle.py
│  │  ├─ setup_wsl_libs.sh
│  │  └─ run_trt_wsl.sh
│  ├─ tests/
│  │  ├─ test_dll.py
│  │  ├─ test_embedding_similarity.py
│  │  ├─ test_tensorrt_embedding.py
│  │  └─ test_reranker_tokenizer_unicode.py
│  ├─ requirements.txt
│  └─ data/
├─ docs/
│  ├─ RAG_DESIGN.md
│  ├─ PDF_PARSER_LOGIC.md
│  └─ STUDY_GUIDE.md
└─ PROJECT_LOG.md
```

### What each boundary means

#### `frontend/`
This is the entire user interface. It is intentionally small. The main logic is almost entirely concentrated in `ChatInterface.jsx`.

#### `backend/main.py`
This is the live API surface. The AGENTS guide says the API surface is only here, and the code confirms that with only `GET /` and streaming `POST /chat` (`backend/main.py:1042-1059`).

#### `backend/core/`
This contains cross-cutting runtime utilities:

- `setup_env.py`: platform-sensitive CUDA / TensorRT / DLL / shared-library setup (`backend/core/setup_env.py:7-176`)
- `utils.py`: query rewrite and HyDE generation helpers (`backend/core/utils.py:9-243`)
- `embedding_shared.py`: shared queue primitive for embedding jobs (`backend/core/embedding_shared.py:3-7`)
- `ollama_runtime.py`: runtime endpoint fallback logic, especially important for WSL-to-host networking (`backend/core/ollama_runtime.py:162-344`)

#### `backend/services/`
This is where the application's heavy logic lives:

- `vector_store.py`: retrieval, expansion, reranking orchestration (`backend/services/vector_store.py:22-533`)
- `embedding_service.py`: singleton embedding runtime using ONNX Runtime + TensorRT (`backend/services/embedding_service.py:39-434`)
- `reranker_service.py`: singleton reranker runtime with richer fallback logic (`backend/services/reranker_service.py:29-1048`)

#### `backend/parsers/`
This contains PDF parsing and chunking logic. The active parser is `PDFLegalParserV2` in `backend/parsers/pdf_parser.py` (`backend/parsers/pdf_parser.py:8-455`).

#### `backend/scripts/`
These are operational tools rather than request-serving code. The most important one is `ingest_pdfs.py`, because it builds the searchable legal corpus (`backend/scripts/ingest_pdfs.py:276-611`).

#### `docs/` and historical notes
These explain intent and historical design direction, but they are not the runtime source of truth.

---

## 4. Entire project architecture in words

If you had to explain the whole repo to someone without showing them code, the cleanest description is this:

> This project is a two-stage legal assistant system. First, it builds a searchable legal knowledge base from ordinance PDFs by parsing them into sections and chunks, embedding those chunks, and storing vectors plus metadata in Pinecone. Second, it serves a chat UI where a user's question is optionally transformed into retrieval queries, relevant legal sections are fetched and reranked, then those sections are inserted into an LLM prompt and streamed back as an answer with source references.

That sentence hides a lot of complexity, so the rest of this section expands it.

### Stage A: Build the legal corpus

The backend does not ship with answers. It ships with a **pipeline for manufacturing retrieval context**.

That pipeline does the following:

1. find target ordinance PDFs under `backend/data/pdfs`
2. parse the PDFs with `unstructured.partition.pdf`
3. group extracted elements into legal sections
4. optionally perform semantic paragraph-based chunking
5. send chunk text into a centralized embedding worker
6. validate the returned vectors
7. upsert vectors plus legal metadata into Pinecone

The most important architectural choice here is that parsing and embedding are not treated as the same kind of work.

- parsing is document/layout work
- embedding is GPU runtime work

That separation is what eventually leads to the worker/queue architecture.

### Stage B: Serve legal chat

When the user asks a question, the backend does not always immediately retrieve legal context. It first asks a model whether retrieval is needed. In `generate_chat_responses`, the backend builds a base system prompt, reconstructs conversation memory, and then performs a first LLM call with a tool definition called `search_legal_database` (`backend/main.py:926-936`).

If the model decides search is needed, the backend retrieves context. If not, it answers directly (`backend/main.py:942-991`).

When retrieval is needed, the backend currently prefers:

1. multi-HyDE passage generation if enabled
2. otherwise single HyDE embedding generation
3. then section retrieval through `VectorStoreManager`
4. then final generation through the Ollama chat stream
5. then source usefulness scoring and reference emission

This means the system is not a trivial “search top-k and paste results” RAG app. It has:

- tool gating
- memory compaction
- HyDE-based query enrichment
- section expansion
- reranking
- streamed answer output
- streamed references near the end

### Stage C: Frontend rendering and session continuity

The frontend creates a persistent session id in local storage and sends it with every chat request (`frontend/src/components/ChatInterface.jsx:7-18`, `frontend/src/components/ChatInterface.jsx:67-71`).

That session id matters because the backend rejects the default session id and uses real session ids to maintain conversation memory (`backend/main.py:903-914`).

This is a good example of the repo's style: seemingly simple UI code and seemingly simple backend code are actually tightly coupled by a small but strict contract.

---

## 5. End-to-end system walkthrough

Here is the whole system in pseudocode first.

```python
def whole_system():
    # offline / operational path
    for ordinance_pdf in backend_data_pdfs:
        sections = parse_pdf_into_sections(ordinance_pdf)
        chunks = chunk_sections(sections)
        vectors = embedding_worker.embed(chunks)
        pinecone.upsert(vectors, metadata=chunks.metadata)

    # online / user path
    user_message = frontend.collect_input()
    session_id = frontend.get_or_create_session_id()
    sse_stream = backend.chat(user_message, session_id)

    backend:
        rebuild_memory(session_id)
        if llm_requests_search(user_message):
            sections = retrieve_relevant_sections(user_message)
            answer = generate_answer_with_context(sections, memory)
            stream(answer_chunks)
            stream(references)
        else:
            answer = generate_answer_without_search(memory)
            stream(answer_chunks)
```

That is the whole system conceptually, but the real details matter.

### Real online path

1. Frontend sends `POST /chat` with `message`, `language`, and `session_id` (`frontend/src/components/ChatInterface.jsx:62-72`).
2. Backend validates session id and initializes conversation memory (`backend/main.py:903-917`).
3. Backend asks DeepSeek whether the search tool should be called (`backend/main.py:926-936`).
4. If search is needed, backend generates HyDE or multi-HyDE retrieval inputs (`backend/main.py:949-966`).
5. Backend retrieves sections and builds context + reference objects (`backend/main.py:968-985`).
6. Backend streams answer chunks through SSE using Ollama (`backend/main.py:993-1001`).
7. Backend appends assistant answer to memory and may compact memory (`backend/main.py:1002-1009`).
8. Backend scores source usefulness and emits references near the end (`backend/main.py:1011-1034`).
9. Frontend incrementally appends `answer` chunks and later attaches `references` to the same assistant message (`frontend/src/components/ChatInterface.jsx:101-145`).

### Real offline/ingest path

1. `ingest_pdfs.py` sets up CUDA/TensorRT environment before heavy imports (`backend/scripts/ingest_pdfs.py:16-25`).
2. It starts a background embedding worker thread (`backend/scripts/ingest_pdfs.py:302-304`).
3. Each target Cap is parsed via `PDFLegalParserV2` (`backend/scripts/ingest_pdfs.py:397-421`).
4. The parser itself may enqueue semantic embedding jobs for paragraph chunking (`backend/parsers/pdf_parser.py:61-116`, `backend/parsers/pdf_parser.py:260-267`).
5. Ingestion enqueues chunk embedding jobs for final vector creation (`backend/scripts/ingest_pdfs.py:446-507`).
6. Worker batches jobs, embeds them, and replies through per-job reply queues (`backend/scripts/ingest_pdfs.py:40-170`).
7. Ingestion validates vectors and upserts to Pinecone (`backend/scripts/ingest_pdfs.py:514-571`).

---

## 6. Frontend architecture

The frontend is intentionally compact, so you can understand most of it by reading four files:

- `frontend/src/main.jsx`
- `frontend/src/App.jsx`
- `frontend/src/components/ChatInterface.jsx`
- `frontend/src/components/ReferenceCard.jsx`

### `frontend/src/main.jsx`

This is the true entrypoint. It imports CSS, imports `App`, and renders the root React tree with `createRoot(...).render(...)` (`frontend/src/main.jsx:1-10`).

It is simple, but it establishes the top-level rule that the frontend is a plain React SPA with no additional state framework.

### `frontend/src/App.jsx`

`App.jsx` holds the dark mode state and toggles the `dark` class on `document.documentElement` via `useEffect` (`frontend/src/App.jsx:4-23`).

Architecturally, `App` is a very thin shell. It does not own chat state. It only owns app-shell state and passes dark mode controls into `ChatInterface`.

### `frontend/src/components/ChatInterface.jsx`

This is the frontend's real center of gravity.

#### Session handling

The frontend defines a fixed local-storage key:

```js
const SESSION_STORAGE_KEY = 'hk-legal-chatbot-session-id';
```

and creates a session id like:

```js
session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}
```

if one does not already exist (`frontend/src/components/ChatInterface.jsx:7-18`).

This is directly aligned with the backend's strict SSE contract note in `AGENTS.md:74-80` and the backend's rejection of `session_id="default"` (`backend/main.py:903-914`).

#### UI state

`ChatInterface` owns:

- current input
- loading state
- language toggle state
- the entire message list

(`frontend/src/components/ChatInterface.jsx:20-31`)

This means the frontend has **local state only**. No Redux, no context, no server-state library.

#### Request format

When the user sends a message, the frontend does a fetch to `http://localhost:8000/chat` with JSON body:

- `message`
- `language`
- `session_id`

(`frontend/src/components/ChatInterface.jsx:62-72`)

That exactly matches the backend request model `ChatRequest` with fields `message`, `language`, and `session_id` (`backend/main.py:435-439`).

#### Streaming parser behavior

The frontend reads the response body as a `ReadableStream`, decodes bytes into text, splits by newline, and only processes lines starting with `data: ` (`frontend/src/components/ChatInterface.jsx:78-106`).

That detail matters a lot. If the backend changed stream formatting, the frontend would silently stop understanding it.

The parser then expects one of three payload shapes:

- `{"answer": "..."}`
- `{"references": [...]}`
- `{"error": "..."}`

(`frontend/src/components/ChatInterface.jsx:105-145`)

This is exactly the contract called out in the repo AGENTS file (`AGENTS.md:74-80`).

#### Why the assistant message is created empty first

The frontend appends an empty assistant message with id `streaming-msg` before reading the stream (`frontend/src/components/ChatInterface.jsx:82-88`). Then:

- each `answer` chunk appends text to `accumulatedContent`
- each `references` event mutates the same streaming message
- after the stream ends, the temporary id is replaced with a timestamp-based id

(`frontend/src/components/ChatInterface.jsx:121-145`)

That design is the frontend half of the backend's “answer chunks first, references later” contract.

### `frontend/src/components/ReferenceCard.jsx`

This component renders a clickable legal citation card with title, citation, type, and external link (`frontend/src/components/ReferenceCard.jsx:4-40`).

One subtle detail: it destructures `page`, but the backend's live reference objects currently carry `pages` arrays rather than a single `page` field (`backend/main.py:828-840`). So the reference card mostly lives off title/citation/source_url/type and only conditionally shows a page if one is directly provided.

This is a good example of how historical UI ideas and current backend structures can partially diverge without fully breaking the user experience.

### Styling layer

The app uses Tailwind through `@import "tailwindcss"` plus a project config reference in `frontend/src/index.css:1-4`, and markdown prose styling overrides in `frontend/src/index.css:5-14`.

The frontend toolchain is intentionally minimal:

- Vite + React plugin (`frontend/vite.config.js:1-7`)
- ESLint flat config (`frontend/eslint.config.js:1-29`)
- no test script and no typecheck script in `frontend/package.json:6-35`

---

## 7. Backend chat flow

The backend's live API surface is small, but the logic behind `/chat` is not.

### API surface

The FastAPI app defines:

- `POST /chat` returning `StreamingResponse(..., media_type="text/event-stream")` (`backend/main.py:1042-1052`)
- `GET /` health endpoint (`backend/main.py:1057-1059`)

That is the entire runtime API.

### Memory model

Conversation state is stored in a `ConversationMemory` dataclass with:

- `summary`
- `turns`
- `summarized_upto`
- `lock`
- `stream_lock`

(`backend/main.py:199-205`)

The backend stores these per session id in a process-local dictionary (`backend/main.py:208-223`).

The design idea is simple:

- `memory.lock` protects conversation state mutations
- `memory.stream_lock` serializes concurrent streams per session

This means the backend intentionally avoids multiple overlapping live generations for the same session.

### Why `session_id="default"` is rejected

The backend normalizes the session id, and if the normalized result is `default`, it emits an SSE error saying a real session id is required for persistent conversation memory (`backend/main.py:903-914`).

This rule is strict because otherwise many browser sessions would collapse into a shared memory bucket and contaminate one another's state.

### Tool-gated retrieval

The most important architectural choice in the chat path is this:

the backend first lets DeepSeek decide whether to call the search tool (`backend/main.py:926-936`).

This means the system is not “always RAG”. It is “RAG when the model decides the user needs legal retrieval.”

If the tool is called, the backend:

1. extracts the search query from the tool call (`backend/main.py:942-945`)
2. imports `get_embedding_service` lazily (`backend/main.py:947`)
3. tries multi-HyDE if enabled (`backend/main.py:949-960`)
4. falls back to single HyDE if needed (`backend/main.py:961-966`)
5. builds context and references (`backend/main.py:968-985`)

If the tool is not called, it answers directly from the base system prompt and memory (`backend/main.py:986-991`).

### Streaming answer path

Once final messages are assembled, the backend converts them to Ollama message format and streams chunks from `stream_ollama_chat(...)` (`backend/main.py:993-1001`).

Each emitted chunk becomes:

```json
{"answer": "chunk_text"}
```

wrapped as an SSE `data:` line (`backend/main.py:996-1000`).

After streaming is done, the backend stores the full answer back into memory (`backend/main.py:1002-1009`).

If sections were retrieved, it then evaluates source usefulness and emits one final SSE event with `references` (`backend/main.py:1011-1034`).

### SSE contract summary

This is the live contract, as established by both code and repo notes:

- request JSON must include `message`, `language`, `session_id` (`backend/main.py:435-439`, `frontend/src/components/ChatInterface.jsx:67-71`)
- response type is `text/event-stream` (`backend/main.py:1045-1052`)
- stream emits `answer` chunks first (`backend/main.py:996-1000`)
- stream emits `references` near the end (`backend/main.py:1033-1034`)
- error conditions may also be emitted as SSE `error` payloads (`backend/main.py:905-913`, `backend/main.py:1037-1039`)

That contract is brittle in a useful way: the pieces are small, but each one is exact.

---

## 8. Retrieval pipeline

The retrieval system is centered around `VectorStoreManager` in `backend/services/vector_store.py`.

### The mental model

The retrieval pipeline is not “retrieve chunks and stop.” It is:

1. generate a retrieval-oriented query representation
2. search Pinecone for candidate chunks
3. expand candidate chunks back to full sections
4. flatten expanded sections if reranking is needed
5. rerank flat chunks
6. regroup reranked chunks back into section objects

That is the real shape of the retrieval system.

### Why `VectorStoreManager` exists

`VectorStoreManager` packages together four concerns:

1. Pinecone connectivity and index shape (`backend/services/vector_store.py:24-101`)
2. embedding service access (`backend/services/vector_store.py:103-111`)
3. query and vector validation (`backend/services/vector_store.py:113-180`)
4. retrieval strategies (`backend/services/vector_store.py:229-533`)

### Index invariants

At initialization, the vector store manager enforces:

- API key must exist or the manager becomes effectively inert (`backend/services/vector_store.py:24-29`)
- index dimension must match expected embedding dimension (`backend/services/vector_store.py:85-94`)
- embedding precision regime is checked against sample vectors (`backend/services/vector_store.py:127-180`)

That precision regime check is extremely important. It is the code's answer to a subtle retrieval problem: if you mix different embedding precision regimes in one index, search ranking can become unstable enough that the retriever is no longer semantically trustworthy.

### Upsert shape

Chunks are upserted with metadata such as:

- `doc_id`
- `section_id`
- `section_title`
- `page_number`
- `chunk_index`
- `total_chunks_in_section`
- `citation`
- `source_url`
- `embedding_precision`
- `embedding_dimension`
- `embedding_strict_fp16`

(`backend/services/vector_store.py:182-228`)

This metadata is what makes full-section expansion possible later.

### Simple retrieval path

`search(query, k)` performs pure embedding similarity search with the asymmetric query prefix, validates the query vector, and calls Pinecone directly (`backend/services/vector_store.py:408-440`).

### Expansion path

`search_with_expansion(query, k)` does similarity search first, then calls `_expand_to_sections(...)` (`backend/services/vector_store.py:229-243`).

`_expand_to_sections(...)` takes retrieved chunks, finds unique `(doc_id, section_id)` pairs, then issues filtered Pinecone queries to fetch sibling chunks for each section (`backend/services/vector_store.py:314-361`).

This is one of the repo's key architectural ideas:

> retrieval rank is done at chunk level, but legal generation context is reconstructed at section level.

That is why the metadata design matters so much.

### HyDE path

The live chat path uses HyDE-centered retrieval. In `backend/main.py`, the backend tries multi-HyDE first, then falls back to single-HyDE (`backend/main.py:949-966`).

On the vector-store side:

- `search_hyde_with_rerank_and_expansion(...)` performs vector search using a precomputed HyDE embedding, then expands sections, flattens them, reranks them, and regroups them (`backend/services/vector_store.py:268-312`).
- `search_multi_hyde_with_rerank_and_expansion(...)` batch-embeds multiple hypothetical passages, runs one Pinecone query per embedding, merges raw hits, expands sections once, reranks, and regroups (`backend/services/vector_store.py:447-533`).

### Retrieval pipeline pseudocode

```python
def retrieval_pipeline(user_query):
    rewritten_or_hypothetical_query = build_retrieval_input(user_query)

    if using_multi_hyde:
        passages = generate_multiple_hypothetical_passages(user_query)
        raw_hits = search_pinecone_for_each(passages)
        merged_hits = deduplicate(raw_hits)
    else:
        hyde_vector = generate_hyde_embedding(user_query)
        merged_hits = search_pinecone(hyde_vector)

    expanded_sections = expand_hits_to_full_sections(merged_hits)
    flat_chunks = flatten(expanded_sections)
    reranked = rerank(user_query, flat_chunks)
    return regroup_into_sections(reranked)
```

### How the docs compare to current code

`docs/RAG_DESIGN.md:90-128` describes a very similar pipeline at a conceptual level. The current live code goes beyond that design document in two ways:

1. it now uses tool-gated retrieval rather than unconditional retrieval
2. it includes multi-HyDE and source usefulness scoring in the live path

So the design doc is still useful, but the code is richer than the old pseudocode.

---

## 9. Ingestion pipeline

The ingestion path is where the repo becomes operationally interesting.

### High-level ingest pipeline

The main async function is `ingest_legal_pdfs(...)` (`backend/scripts/ingest_pdfs.py:276-611`). Its own docstring already states the five stages:

1. scan PDF directory
2. start background embedding worker
3. parse PDFs
4. generate embeddings
5. upsert to Pinecone

### Why environment setup is first

At the very top of the script, before importing heavy runtime services, the script does:

- `setup_env.setup_cuda_dlls()` (`backend/scripts/ingest_pdfs.py:16-25`)

This is not an implementation detail. It is a hard invariant repeated across the repo and documented in `AGENTS.md:66-68`.

The reason is simple: by the time `torch` or `onnxruntime` imports happen, the runtime must already be able to see the relevant CUDA/TensorRT libraries.

### Why the script sets a larger TensorRT workspace

Before service import, the ingestion script sets `EMBEDDING_TRT_MAX_WORKSPACE_SIZE` to `6 * 1024**3` if not already set (`backend/scripts/ingest_pdfs.py:18-20`).

This tells you something important about ingest versus chat:

- chat latency wants responsiveness
- ingestion throughput wants larger one-time optimization space

So the project intentionally tunes ingestion for a more throughput-oriented TensorRT regime.

### Parse stage

For each Cap, `process_cap(cap_num)` either loads an existing parsed JSON or constructs `PDFLegalParserV2` and calls `process_ordinance(...)` (`backend/scripts/ingest_pdfs.py:397-421`).

The parser:

1. calls `partition_pdf(...)` from `unstructured.partition.pdf` (`backend/parsers/pdf_parser.py:121-132`)
2. groups elements into logical sections (`backend/parsers/pdf_parser.py:172-199`)
3. linearizes and paragraph-splits section content (`backend/parsers/pdf_parser.py:202-258`)
4. optionally embeds paragraphs for semantic chunking (`backend/parsers/pdf_parser.py:260-267`)
5. merges semantically similar paragraphs into chunks with overlap (`backend/parsers/pdf_parser.py:281-375`)
6. filters degenerate header-only chunks (`backend/parsers/pdf_parser.py:414-435`)
7. annotates total chunks per section and saves JSON (`backend/parsers/pdf_parser.py:437-455`)

### Embedding stage

After parsing, ingestion prefixes each chunk with the document retrieval prefix and then divides work into embedding batches (`backend/scripts/ingest_pdfs.py:433-507`).

Importantly, ingestion does **not** embed directly in the Cap-processing thread. Instead it enqueues work to a shared worker.

### Validation and upsert stage

Returned vectors are validated for:

- dimension
- finite numeric values
- non-near-zero norm

(`backend/scripts/ingest_pdfs.py:201-213`, `backend/scripts/ingest_pdfs.py:519-557`)

Only valid vectors are upserted, in Pinecone batch sizes of 100 (`backend/scripts/ingest_pdfs.py:559-571`).

### Ingestion pipeline pseudocode

```python
async def ingest_legal_pdfs(caps):
    setup_gpu_runtime_before_imports()
    start_embedding_worker_thread()
    verify_tensorrt_is_active()
    warm_tensorrt_cache_if_needed()

    for cap in caps:
        chunks = parse_or_load_json(cap)
        batch_jobs = split_chunks_for_embedding(chunks)
        enqueue_embedding_jobs(batch_jobs)
        vectors = collect_worker_replies(batch_jobs)
        vectors = validate(vectors)
        pinecone.upsert(vectors)

    stop_worker_and_wait_for_queue_to_drain()
```

### Why this stage became hard

Because ingest combines three kinds of work that fail differently:

1. PDF/layout extraction work
2. GPU inference work
3. remote vector-store writes

If you treat them as one monolithic synchronous operation, failures become harder to isolate and GPU runtime instability becomes much more expensive.

That is exactly why the worker/queue architecture matters.

---

## 10. Worker and queue architecture

This is the most important part of the repo to understand if you want to understand the ingestion failure and the later stabilization work.

### The shared queue primitive

`backend/core/embedding_shared.py` is tiny:

```python
job_q = Queue(maxsize=32)
STOP_TOKEN = None
```

(`backend/core/embedding_shared.py:3-7`)

The file is small, but the architectural meaning is large:

there is one shared embedding-work channel used across the ingestion system.

### Two producers, one embedding runtime

There are effectively **two producers** of embedding requests:

1. the parser, when semantic chunking is enabled and paragraph similarity needs embeddings (`backend/parsers/pdf_parser.py:61-116`, `backend/parsers/pdf_parser.py:260-267`)
2. the ingestion process itself, when final chunk embeddings are needed for Pinecone upload (`backend/scripts/ingest_pdfs.py:446-507`)

That means the queue architecture is not just a convenience. It is the coordination point between:

- document chunking logic
- final vectorization logic
- the single live embedding runtime

### The worker thread

`embedding_worker()` is started as a daemon thread at the beginning of ingestion (`backend/scripts/ingest_pdfs.py:302-304`).

The worker immediately grabs the singleton embedding service once:

```python
service = get_embedding_service()
```

(`backend/scripts/ingest_pdfs.py:45-47`)

This is the design's core idea:

> load the embedding runtime once, then reuse it for queued work rather than repeatedly loading model/runtime state from many concurrent paths.

That is the answer to your “I think it's because the embedding model can only be loaded once at a time” intuition: yes, that is very close to the real design motivation.

More precisely, the current code is optimized around these facts:

- `EmbeddingService` is a singleton (`backend/services/embedding_service.py:39-61`)
- model bootstrap is protected by `_bootstrap_lock` (`backend/services/embedding_service.py:70-105`, `backend/services/embedding_service.py:113-121`)
- embedding calls themselves are serialized through `_embed_lock` (`backend/services/embedding_service.py:286-296`)

So even if many threads or logical producers want embeddings, the actual runtime wants to behave like one controlled shared service, not many independent model instances racing for GPU state.

### Micro-batching design

The worker does not process one job at a time if it can avoid it. It tries to drain more jobs for a short time window and merge them into a micro-batch (`backend/scripts/ingest_pdfs.py:58-98`).

This helps because:

- batching usually improves GPU throughput
- a single service call is cheaper than many tiny calls
- the queue can combine parser-originated and ingest-originated requests

The worker then uses an internal `_embed_with_fallback(...)` helper that retries the merged texts in smaller sub-batch sizes if needed (`backend/scripts/ingest_pdfs.py:104-131`).

So there are actually **two batching layers**:

1. queue-level micro-batching across jobs
2. worker-level sub-batch fallback within the merged workload

That is a very practical design for unstable or memory-sensitive GPU inference.

### Reply path

Each queued job carries its own `reply_q` (`backend/scripts/ingest_pdfs.py:145-161`, `backend/scripts/ingest_pdfs.py:455-478`, `backend/parsers/pdf_parser.py:64-83`).

This means the architecture is not a plain fire-and-forget work queue. It is a request/reply queue system:

- producer enqueues `{id, texts, reply_q, source}`
- worker embeds
- worker slices merged vectors back into per-job payloads
- worker writes result into the job's reply queue
- producer blocks waiting for its own reply queue

### Worker architecture pseudocode

```python
job_q = Queue(maxsize=32)

def producer(texts, source):
    reply_q = Queue(maxsize=1)
    job_q.put({"type": "embed_request", "texts": texts, "reply_q": reply_q, "source": source})
    return reply_q.get(timeout=...)

def embedding_worker():
    service = get_embedding_service()  # load once, reuse
    while True:
        job = job_q.get()
        if job is STOP_TOKEN:
            break

        jobs = microbatch(job_q, first_job=job)
        merged_texts = merge(jobs)
        merged_vectors = embed_in_subbatches(service, merged_texts)

        for original_job in jobs:
            slice_for_job = take_matching_vector_slice(merged_vectors)
            original_job.reply_q.put({"vectors": slice_for_job})
```

### Why this architecture was introduced

Based on the current executable code plus the historical plans in `.sisyphus/plans/fix-embedding-deadlock.md` and `.sisyphus/plans/fix-tensorrt-cuda-runtime-error.md`, the repo had to solve several real problems:

1. repeated model/runtime initialization is expensive and fragile
2. concurrent embedding access can deadlock or compete for GPU/runtime resources
3. TensorRT errors often become worse under uncontrolled concurrency
4. some workloads need smaller fallback sub-batches after failures

So the worker/queue design solves two classes of problem at once:

- **correctness/stability**: centralize access to one runtime
- **performance**: batch requests together instead of treating each small request independently

### The user's intuition, refined

Your explanation was directionally right:

> “the embedding model can only be loaded once at a time so we reuse that session instead of loading multiple sessions”

The more exact version is:

> the repo converged on a singleton embedding runtime plus queue-based access because model bootstrap, TensorRT engine state, GPU memory use, and thread safety all become easier to control when one process-wide service owns embedding inference and many producers submit work to it instead of creating competing runtimes.

That is a better mental model than “the model literally cannot be loaded twice.” It is more that the project treats multiple concurrent model lifecycles as a source of instability and waste.

---

## 11. Embedding runtime architecture

`EmbeddingService` is one of the most important files in the repo.

### Singleton design

The class is explicitly a singleton with:

- `_instance`
- `_init_lock`
- `_bootstrap_lock`
- instance-level `_embed_lock`

(`backend/services/embedding_service.py:39-66`)

That detail matters historically because one of the internal plans documents a past deadlock where singleton initialization and embedding operations shared the same lock (`.sisyphus/plans/fix-embedding-deadlock.md:20-36`).

The current code shows that this has been corrected by separating initialization locking from embedding-operation locking.

### Load-once behavior

`ensure_loaded()` loads the model only once per process (`backend/services/embedding_service.py:113-121`).

That bootstrap step includes:

- tokenizer load (`backend/services/embedding_service.py:162-169`)
- max-length setup (`backend/services/embedding_service.py:171-182`)
- ONNX Runtime session configuration (`backend/services/embedding_service.py:184-249`)
- active provider verification (`backend/services/embedding_service.py:251-273`)

### Provider policy

For embeddings, the current code strongly prefers **TensorRT** and explicitly rejects a live `CUDAExecutionProvider` path.

If TensorRT is required but unavailable, `_load_model()` raises immediately (`backend/services/embedding_service.py:192-201`).

If active providers include `CUDAExecutionProvider`, it also raises (`backend/services/embedding_service.py:267-271`).

That makes the embedding path very opinionated:

- preferred: `TensorrtExecutionProvider`
- allowed fallback: CPU only if TensorRT requirement is disabled
- explicitly disallowed: CUDA EP in the active provider chain

This is also reflected in the dedicated test script, which fails if CUDA EP is active during the TensorRT embedding test (`backend/tests/test_tensorrt_embedding.py:37-45`).

### Why the repo is so strict here

Because the embedding runtime is treated as the most performance-sensitive and instability-prone part of ingestion. The code wants:

- a known provider mode
- predictable precision regime
- cached TensorRT engines
- controlled threading

That is why the service exposes knobs such as:

- `EMBEDDING_REQUIRE_TENSORRT`
- `EMBEDDING_TRT_FP16`
- `EMBEDDING_STRICT_FP16`
- `EMBEDDING_TRT_MAX_WORKSPACE_SIZE`
- `EMBEDDING_TRT_MIN_SUBGRAPH_SIZE`
- `EMBEDDING_TRT_MAX_PARTITION_ITERATIONS`
- `EMBEDDING_TRT_AUX_STREAMS`

(`backend/services/embedding_service.py:78-111`)

### Inference behavior

Each embedding call:

1. ensures model is loaded (`backend/services/embedding_service.py:290`)
2. normalizes blank text to `.` to avoid crashes (`backend/services/embedding_service.py:292-293`)
3. serializes inference with `_embed_lock` (`backend/services/embedding_service.py:295-296`)
4. tokenizes input (`backend/services/embedding_service.py:317-324`)
5. injects `position_ids` if missing (`backend/services/embedding_service.py:326-332`)
6. runs ONNX model inference (`backend/services/embedding_service.py:334-336`)
7. mean-pools outputs (`backend/services/embedding_service.py:338-346`)
8. normalizes vectors (`backend/services/embedding_service.py:348-352`)
9. validates non-zero norm and dimension correctness (`backend/services/embedding_service.py:354-385`)

That “position ids” step is easy to miss, but it is a clue that the exported model graph and the tokenizer/runtime expectations needed manual coordination.

### Error recovery behavior

If validation failures happen in FP16 mode and strict mode is not enabled, the service can demote itself to FP32 TensorRT (`backend/services/embedding_service.py:275-284`, `backend/services/embedding_service.py:397-399`).

If transient TensorRT-like errors such as `enqueueV3`, `TensorRT EP`, or `Cuda Runtime` occur, the service retries with exponential backoff (`backend/services/embedding_service.py:401-413`).

If those failures persist, it clears the TensorRT cache and raises (`backend/services/embedding_service.py:416-423`).

This is a direct sign that the project ran into real engine/runtime instability, not hypothetical instability.

---

## 12. Reranker runtime architecture

The reranker is similar in spirit to the embedding service but more flexible in its provider strategy.

### Why the reranker differs from the embedding path

`RerankerService` supports a much richer fallback ladder. Its `_build_provider_candidates()` method constructs candidate provider chains including:

- strict TensorRT-only attempts
- experimental TensorRT -> CUDA -> CPU chains
- CUDA -> CPU fallback
- CPU-only fallback

(`backend/services/reranker_service.py:188-365`)

This is a much more tolerant design than the embedding service.

### Why that likely happened

The most plausible explanation, based on current code and supporting evaluation scripts, is:

- embeddings are the central ingestion bottleneck and need one highly controlled mode
- reranking is important, but the system is more willing to accept reduced provider performance to preserve overall system correctness

This is visible in `backend/llm_evaluate.py`, which can force CPU reranker mode by default for evaluation (`backend/llm_evaluate.py:1187-1213`), and in `backend/scripts/verify_cpu_rerank_pipeline.py`, which explicitly sets:

- `RERANKER_FORCE_CPU=1`
- `RERANKER_REQUIRE_TENSORRT=0`

(`backend/scripts/verify_cpu_rerank_pipeline.py:125-137`)

### Core reranker behavior

The reranker:

- loads tokenizer and ONNX session (`backend/services/reranker_service.py:520-599`)
- may detect either a classification architecture or causal-LM reranker mode (`backend/services/reranker_service.py:641-688`)
- formats query/document pairs, tokenizes them, executes ONNX inference, extracts scores, and sorts documents (`backend/services/reranker_service.py:690-1006`)

It also normalizes Unicode-heavy text before reranking (`backend/services/reranker_service.py:1009-1016`), which is exactly why there is a dedicated Unicode reranker test (`backend/tests/test_reranker_tokenizer_unicode.py:30-64`).

---

## 12A. Fundamental concepts from zero

This section is for a reader who does **not** already know the ML/runtime terms used in this repo.

If words like embedding, inference, ONNX, TensorRT, CUDA, worker, queue, or VRAM feel abstract, read this section first.

### What is this project doing in plain language?

This project is trying to answer Hong Kong legal questions using real legal documents.

To do that, it needs two separate abilities:

1. it needs to **build a searchable knowledge base** from ordinance PDFs
2. it needs to **search that knowledge base** when a user asks a question

That is why the repo has two big stories:

- **ingestion**: turn PDFs into searchable vectors
- **chat**: retrieve relevant legal text and answer questions with it

### What is an embedding?

An embedding is a list of numbers that represents the meaning of a piece of text.

Think of it as a semantic fingerprint.

For example:

- “I was injured at work”
- “Can I claim compensation for a workplace accident?”

These sentences use different words, but they mean similar things. A good embedding model turns them into vectors that are numerically close.

That is how semantic search works. Instead of searching only for exact words, the system searches for **similar meaning**.

In this repo:

- legal text chunks are embedded and stored in Pinecone
- user queries are embedded too
- Pinecone finds chunks whose vectors are close to the query vector

### What is inference?

This repo is not training models. It is only **running** already-trained models.

Running a trained model to get an output is called **inference**.

Examples in this repo:

- text -> embedding model -> vector
- query + candidate chunk -> reranker -> relevance score

So when this guide talks about GPU runtime, batching, TensorRT, CUDA, ONNX Runtime, or VRAM, it is talking about how the repo performs inference efficiently and reliably.

### What is a reranker?

A reranker is a model that takes:

- a query
- a set of candidate chunks/documents

and scores which candidates are most relevant.

Why not trust the embedding search alone?

Because embedding search is good at quickly finding a shortlist, but it is not always the most precise final judge. So a common retrieval pattern is:

1. embeddings retrieve a shortlist fast
2. reranker reorders that shortlist more accurately

That is also the pattern used in this repo (`backend/services/vector_store.py:245-266`, `backend/services/vector_store.py:301-312`, `backend/services/reranker_service.py:876-1006`).

### What is ONNX?

ONNX is a model format.

You can think of it as a portable exported version of a trained model. Instead of only running the model in its original framework, the model is exported into ONNX form so another runtime can execute it.

That is why this repo contains export scripts for embedding and reranker models:

- `backend/scripts/export_yuan_trt.py:16-73`
- `backend/scripts/export_yuan_cuda.py:1-24`
- `backend/scripts/export_reranker_onnx.py:93-297`
- `backend/scripts/export_zerank_onnx.py:405-595`

### What is ONNX Runtime?

ONNX Runtime is the software that loads and runs ONNX models.

It is not the model itself. It is the runtime engine.

A simple mental model is:

- ONNX file = the blueprint
- ONNX Runtime = the machine that runs the blueprint

In this repo:

- embeddings use ONNX Runtime through `ORTModelForFeatureExtraction` (`backend/services/embedding_service.py:35`, `backend/services/embedding_service.py:243-249`)
- reranking uses ONNX Runtime sessions more directly (`backend/services/reranker_service.py:374-507`)

### What is an execution provider?

ONNX Runtime can delegate actual execution to different backends. Those backends are called **execution providers**.

Examples:

- `CPUExecutionProvider`
- `CUDAExecutionProvider`
- `TensorrtExecutionProvider`

So ONNX Runtime is the host runtime, and the provider is the actual backend doing the work.

### What is CUDA?

CUDA is NVIDIA's GPU computing platform and runtime layer.

The shortest useful explanation is:

> CUDA is the foundation that lets software use an NVIDIA GPU for computation.

It is lower-level than TensorRT.

### What is TensorRT?

TensorRT is NVIDIA's inference optimization and execution engine.

The shortest useful explanation is:

> TensorRT is a specialized high-performance engine for running neural-network inference on NVIDIA GPUs.

It tries to run the model faster and more efficiently than a more generic execution path.

### What is the difference between CUDA and TensorRT?

This is the easiest way to remember it:

- **CUDA** = the general GPU compute foundation
- **TensorRT** = a specialized inference engine built for neural networks

Another way to say it:

- CUDA says: “you can run compute on the GPU”
- TensorRT says: “this is neural-network inference, so I will optimize it aggressively”

### What is VRAM?

VRAM is the memory on the GPU.

It is where the GPU keeps things like:

- model weights
- intermediate tensors
- input/output buffers
- cached memory
- TensorRT engine/context state

If VRAM is exhausted, inference becomes unstable, slow, or impossible.

### Why can VRAM seem to “accumulate”?

When people say “VRAM keeps accumulating,” it does **not always** mean there is a real memory leak.

Sometimes it means:

1. the model is still loaded
2. the runtime is caching memory for reuse
3. TensorRT or PyTorch is holding memory pools/buffers/context state
4. the process is still alive, so the runtime still expects to reuse that memory

So the careful answer is:

> different runtimes show different memory patterns, but both CUDA-backed and TensorRT-backed execution can retain GPU memory depending on caching, pooling, and process lifetime.

### What is batching?

Batching means running multiple inputs together in one model call.

Instead of this:

```python
embed(text_1)
embed(text_2)
embed(text_3)
```

you do this:

```python
embed([text_1, text_2, text_3])
```

Why batch?

- better GPU utilization
- less overhead per item
- higher throughput

But large batches also increase memory pressure. That is why this repo has fallback sub-batch logic instead of only trying one large batch forever (`backend/scripts/ingest_pdfs.py:104-131`).

### What is a worker?

A worker is a dedicated part of the program whose job is to process a certain kind of work.

In this repo, the embedding worker exists so that many parts of ingestion do not all directly hit the embedding runtime at once.

### What is a queue?

A queue is a waiting line for jobs.

In this repo, multiple producers can say:

> “please embed these texts when the shared runtime is ready.”

That allows one controlled runtime to serve many callers.

### Why use a worker + queue here?

Because embeddings in this repo are not cheap helper-function calls. They are expensive GPU inference operations with runtime state, provider assumptions, and memory sensitivity.

The worker/queue design gives the project:

- one shared embedding runtime
- controlled batching
- controlled retries/fallbacks
- less repeated model loading
- clearer error handling

The short version is:

> the queue protects the runtime, and the worker owns the runtime.

### Why WSL mattered

WSL mattered because these GPU inference issues were not only Python issues. They were also native-library and runtime-loading issues.

The repo eventually treated WSL as a serious path because Linux-style shared-library loading and GPU runtime behavior were often easier to control than a mixed Windows-native environment (`backend/core/setup_env.py:7-97`, `backend/core/ollama_runtime.py:173-198`, `backend/README.md:62-103`).

---

## 13. TensorRT, CUDA, and ONNX Runtime explained in the context of this repo

This is the section that answers your technical questions directly.

### What is CUDA?

CUDA is NVIDIA's GPU computing platform and programming/runtime layer. In practical terms for this repo, CUDA is the lower-level environment that makes GPU execution possible.

If you want the shortest useful mental model:

> CUDA is the GPU compute foundation.

It is not the same thing as TensorRT. It is the lower layer that GPU compute libraries and runtimes rely on.

In this repo, CUDA shows up in two ways:

1. through `torch` GPU behavior and cache clearing (`backend/services/embedding_service.py:126-133`)
2. through ONNX Runtime provider availability and TensorRT's dependency on underlying GPU libraries (`backend/core/setup_env.py:7-176`)

The external research also reinforces that CUDA is the platform-level layer while TensorRT is a higher-level inference optimizer/execution engine.

### What is TensorRT?

TensorRT is NVIDIA's optimized inference engine for neural network execution on NVIDIA GPUs.

Shortest useful mental model:

> TensorRT is not “GPU support” in general. It is a specialized inference acceleration engine that tries to run model graphs faster and more efficiently than a more generic execution path.

In this repo, TensorRT is used through ONNX Runtime's `TensorrtExecutionProvider`, not by hand-writing TensorRT engines directly in most runtime code (`backend/services/embedding_service.py:205-249`, `backend/services/reranker_service.py:223-365`).

### What is ONNX Runtime?

ONNX Runtime is the framework that loads and executes ONNX models while letting different execution providers do the actual backend work.

Shortest useful mental model:

> ONNX Runtime is the host runtime, and TensorRT or CUDA are possible execution backends under it.

That is exactly how this repo uses it:

- embeddings: `ORTModelForFeatureExtraction` (`backend/services/embedding_service.py:35`, `backend/services/embedding_service.py:243-249`)
- reranker: direct `onnxruntime.InferenceSession` logic (`backend/services/reranker_service.py:374-507`)

### Relationship between the three

For this repo, the relationship is best understood like this:

```text
Model weights / exported ONNX graph
        ↓
ONNX Runtime
        ↓
Execution Provider (TensorRT or CPU, sometimes CUDA for reranker fallback)
        ↓
CUDA libraries / GPU driver stack
        ↓
NVIDIA GPU
```

So:

- **CUDA** = GPU compute/runtime foundation
- **TensorRT** = specialized optimized inference engine
- **ONNX Runtime** = orchestrating runtime that loads ONNX graphs and delegates execution to providers

### Why use TensorRT here?

Because this repo wants high-throughput inference for embeddings and possibly reranking, especially during ingestion. The project log explicitly records a major ONNX/TensorRT migration for boosted GPU inference (`PROJECT_LOG.md:319-331`).

The embedding service code also makes clear that the intended high-performance path is TensorRT with engine caching, workspace tuning, FP16 preference, and preflight verification (`backend/services/embedding_service.py:188-234`, `backend/scripts/ingest_pdfs.py:216-270`).

### What is the difference between TensorRT and CUDA in practical terms?

For this repo, the practical difference is:

- **CUDA alone** means a more general GPU execution path.
- **TensorRT** means a more aggressively optimized inference path for a model graph.

That is why the embedding test explicitly says TensorRT must be active and CUDA EP must not be active (`backend/tests/test_tensorrt_embedding.py:37-45`).

The repo is effectively saying: “for embeddings, the intended production-quality path is TensorRT, not generic CUDA EP.”

### Why might CUDA appear to accumulate VRAM while TensorRT seems not to?

This needs careful wording.

The safest accurate explanation is:

1. **Both CUDA-based runtimes and TensorRT-based runtimes can hold GPU memory**.
2. What users often describe as “VRAM accumulation” is usually the visible result of:
   - memory pools
   - allocator caching
   - engine caching
   - fragmentation
   - long-lived process reuse
3. In PyTorch and related CUDA runtimes, cached GPU memory often stays reserved for reuse even after a step is done. That can make memory look “stuck” even when it is being intentionally retained for performance.
4. TensorRT may *feel* different because a large part of its work is engine compilation + reusable engine/context behavior rather than the same allocator behavior the user is used to observing in PyTorch.

So the right answer is **not** “CUDA accumulates VRAM but TensorRT never does.” That would be too absolute.

The better answer is:

> CUDA-visible memory growth is often allocator/pool behavior, while TensorRT may shift the memory pattern toward engine/cache/context usage. The observed memory behavior can feel different, but both systems can retain GPU memory depending on how the runtime, allocator, and process lifecycle are managed.

That also matches the repo's own engineering choices:

- explicit TensorRT cache directory and warmup (`backend/services/embedding_service.py:188-190`, `backend/scripts/ingest_pdfs.py:246-270`)
- explicit `torch.cuda.empty_cache()` on unload (`backend/services/embedding_service.py:126-133`)

### Why use both TensorRT and ONNX Runtime together?

Because ONNX Runtime gives the repo a common model-execution interface while TensorRT gives a fast provider backend when available.

The repo does not want to manually own every detail of low-level inference execution. Instead it wants:

- ONNX-exported models
- one runtime API
- provider selection and fallback
- TensorRT acceleration where appropriate

That is exactly what ONNX Runtime execution providers are for.

---

## 14. Windows vs WSL setup and why WSL matters here

This repo clearly documents a split runtime strategy.

### The official repo-level split

The top-level AGENTS file and backend README both say:

- Windows runtime uses `.venv`
- WSL/Linux runtime uses `.venv-wsl`
- do not mix them

(`AGENTS.md:70-72`, `backend/README.md:18-30`, `backend/README.md:62-66`)

This is not just a preference. It is a response to platform-specific dependency resolution and binary compatibility.

### Why the split exists in code, not just docs

`backend/requirements.txt` is explicitly platform-sensitive:

- Windows installs `torch==2.6.0+cu124` and matching `torchaudio` / `torchvision`
- non-Windows installs plain `torch==2.6.0`

(`backend/requirements.txt:693-726`)

The backend README also calls this out directly (`backend/README.md:26-30`).

So the project is already architected around the idea that Windows and WSL are not just different shells. They are different binary/runtime environments.

### How WSL is supported in runtime code

There are two important code signs that WSL is a real first-class concern:

#### 1. TensorRT shared-library preload on Linux/WSL

`backend/core/setup_env.py` has a dedicated `_setup_linux_trt_libs()` path that searches `tensorrt_libs`, adds library paths to `LD_LIBRARY_PATH`, and preloads TensorRT and CUDA shared objects with `ctypes.CDLL(..., mode=RTLD_GLOBAL)` before ONNX Runtime import (`backend/core/setup_env.py:7-97`).

That is a very strong signal that WSL/Linux was not an afterthought. The project needed active runtime help to make TensorRT visible and stable there.

#### 2. WSL-specific Ollama endpoint discovery

`backend/core/ollama_runtime.py` detects WSL and tries gateway IPs, `/etc/resolv.conf` nameserver IPs, and host overrides when resolving candidate Ollama endpoints (`backend/core/ollama_runtime.py:173-198`).

This means the project knows a WSL process may need to talk to a host-side Ollama instance differently from a plain Windows process.

### How to set up WSL for this project

Based on the repo's own docs, the practical WSL setup flow is:

1. open WSL/Linux terminal
2. create a separate venv:

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

(`AGENTS.md:41-47`)

3. if using TensorRT in WSL, ensure TensorRT/ONNX Runtime GPU libs are actually visible. The repo contains supporting scripts and notes for this:

- `backend/scripts/setup_wsl_libs.sh:10-34`
- `backend/scripts/run_trt_wsl.sh:1-25`
- `.sisyphus/drafts/wsl-setup-instructions.md:8-40`

4. if using Ollama from WSL, use `backend/scripts/ollama_discover.py` or the README's guidance to discover the host-reachable endpoint (`backend/README.md:78-103`)

### Why WSL may be more stable than native Windows here

Based on the repo's code and the external runtime research, the clean explanation is:

1. WSL gives the project a Linux-style runtime environment for Python, ONNX Runtime, TensorRT shared libraries, and GPU-facing dependencies.
2. The repo already contains Linux-specific TensorRT library preload logic (`backend/core/setup_env.py:7-97`), which means WSL became part of the intended stable path.
3. The project repeatedly documents and guards against mixed Windows/WSL environments because mixing binary stacks causes path and library issues (`AGENTS.md:70-72`, `backend/README.md:18-18`, `backend/llm_evaluate.py:1323-1331`).
4. For ML inference stacks, Linux-family environments often produce more consistent behavior for GPU libraries, dynamic linking, and runtime tooling than Windows-native combinations of Python packages + DLL resolution.

So the best explanation is:

> WSL was useful not because it magically made the model smarter, but because it gave the project a cleaner Linux-style runtime for ONNX Runtime, TensorRT, and related GPU dependencies, while also avoiding some of the DLL-path and mixed-environment issues that were easier to hit on native Windows.

That is also why the repo keeps separate virtualenvs and warns so strongly about not mixing them.

---

## 15. Postmortem: why the PDF ingestion script failed and what the project changed

This section answers the “what mistake did we make?” question as carefully as possible.

### Short answer

The ingestion path became fragile because it combined:

- heavy PDF parsing
- semantic chunking that itself needed embeddings
- final chunk embedding for upload
- TensorRT-dependent runtime assumptions
- queueing, concurrency, and platform-specific GPU dependencies

all inside one operational workflow.

The fix was not one single patch. It was a gradual architectural convergence toward:

- one shared embedding runtime
- a queue/worker model
- stricter provider checks
- cache warmup and invalidation logic
- better isolation between parsing and embedding work
- better Windows/WSL separation

### Symptom level

The current ingest script's error handling shows the failure signatures the project learned to treat seriously:

- TensorRT not active (`backend/scripts/ingest_pdfs.py:216-233`)
- queue full while waiting to enqueue work (`backend/scripts/ingest_pdfs.py:468-472`)
- timed out waiting for worker reply (`backend/scripts/ingest_pdfs.py:493-499`)
- embedding failure returned from worker (`backend/scripts/ingest_pdfs.py:501-503`)
- persistent embedding runtime failure causing critical stop (`backend/scripts/ingest_pdfs.py:581-594`)

That means the failure was not just “script crashed.” It was more like “the runtime pipeline had too many ways to stall, destabilize, or misconfigure itself under load.”

### The likely mistake chain

Based on executable code plus historical project notes, the main mistakes/pain points appear to have been:

#### Mistake 1: treating embedding like ordinary function work instead of scarce runtime work

The project eventually moved to a dedicated worker plus singleton runtime. That usually happens because direct uncontrolled access caused enough trouble to force centralization.

The existence of:

- singleton embedding service (`backend/services/embedding_service.py:39-66`)
- queue-backed shared access (`backend/core/embedding_shared.py:3-7`)
- worker batching (`backend/scripts/ingest_pdfs.py:58-131`)
- historical deadlock fix plan (`.sisyphus/plans/fix-embedding-deadlock.md:20-36`)

strongly suggests that the project learned this lesson the hard way.

#### Mistake 2: letting runtime/provider assumptions remain implicit

The current script now explicitly verifies TensorRT activation before ingestion proceeds (`backend/scripts/ingest_pdfs.py:216-233`) and warms the cache if needed (`backend/scripts/ingest_pdfs.py:246-270`).

That implies earlier versions were more likely to “hope the runtime is correct” instead of proving it up front.

#### Mistake 3: underestimating platform-specific dependency friction

The repo now repeatedly warns not to mix Windows and WSL virtualenvs (`AGENTS.md:70-72`, `backend/README.md:62-66`).

It also contains dedicated WSL runtime support code (`backend/core/setup_env.py:7-97`) and WSL Ollama fallback logic (`backend/core/ollama_runtime.py:173-198`).

This suggests the project learned that GPU inference issues were sometimes environment issues disguised as model issues.

#### Mistake 4: assuming bigger batches are always better

The worker now has explicit micro-batch and sub-batch fallback tuning (`backend/scripts/ingest_pdfs.py:182-198`, `backend/scripts/ingest_pdfs.py:104-131`).

That is usually the sign of real-world memory or runtime instability under large batches.

#### Mistake 5: mixing historical vectors or precision regimes too casually

The vector store now enforces precision regime compatibility and warns about legacy vectors without precision metadata (`backend/services/vector_store.py:127-180`).

That tells you retrieval quality problems were not only runtime problems. They were also data-regime consistency problems.

### Why the worker/queue architecture solved the problem

The worker/queue model solved the ingestion problem by changing the shape of the work.

Before this kind of architecture, the failure model is usually:

- many callers want embeddings
- they each try to use GPU/runtime resources directly
- model loading, engine state, threading, and memory behavior become harder to predict

After the worker/queue architecture, the system becomes:

- many producers
- one controlled embedding service owner
- one queue for work admission control
- one place to do batching, sub-batch fallback, and error propagation

That is exactly what the current code implements.

### Failure path pseudocode

```python
def bad_ingestion_shape():
    parse_pdf()
    semantic_chunk_embeddings()
    final_chunk_embeddings()
    upload_vectors()
    # all contending for runtime assumptions and GPU state

def improved_ingestion_shape():
    worker = single_embedding_runtime_owner()
    parser.enqueue_semantic_embedding_requests(worker)
    ingest.enqueue_final_embedding_requests(worker)
    worker.microbatches_and_recovers()
    validated_vectors = collect_replies()
    upload(validated_vectors)
```

### Was the core reason “embedding model should only be loaded once”?

That is a good simplified explanation, but the more complete answer is:

- not just one model load
- one controlled **runtime lifecycle**
- one controlled **engine/cache state**
- one controlled **GPU inference entrypoint**

That is why the current system uses both singleton service design and queue-based work submission.

---

## 16. What TensorRT/ONNX export scripts tell us about the project

The export scripts are useful for understanding how seriously the repo treats optimized inference.

### Embedding exports

- `backend/scripts/export_yuan_trt.py` exports the Yuan embedding model to ONNX, then converts to FP16 and writes under `backend/models/yuan-onnx-trt` (`backend/scripts/export_yuan_trt.py:16-73`)
- `backend/scripts/export_yuan_cuda.py` exports a CUDA-ready ONNX version to `backend/models/yuan-onnx-cuda` (`backend/scripts/export_yuan_cuda.py:1-24`)

This tells you the embedding stack was not left as “just run Transformers directly.” The project deliberately invested in model export and ONNX/TensorRT acceleration.

### Reranker exports

- `backend/scripts/export_reranker_onnx.py` is a more defensive FP16 export path for `bge-reranker-v2-m3` with ORT-loadability checks (`backend/scripts/export_reranker_onnx.py:93-297`)
- `backend/scripts/export_zerank_onnx.py` is an even more defensive export path for `zeroentropy/zerank-1-small`, including mixed-type harmonization passes (`backend/scripts/export_zerank_onnx.py:405-595`)

That tells you the project learned a painful lesson here too: exported ONNX artifacts are not automatically safe or production-ready just because they exist. They needed real validation and conversion safeguards.

---

## 17. Historical pivots and the mistakes they reveal

`PROJECT_LOG.md` is not executable source, but it is valuable as a history of why the repo looks the way it does today.

### Pivot 1: from general chat app to legal RAG

The early project started as a simpler chat backend/frontend stack, then moved toward legal retrieval and references (`PROJECT_LOG.md:5-31`, `PROJECT_LOG.md:37-76`).

This reveals the first understandable mistake: the project initially had less retrieval and indexing complexity than it later needed.

### Pivot 2: from scraping to PDF-first legal corpus building

The log shows a pivot from HKLII scraping and dynamic scraping experiments toward e-Legislation PDF ingestion (`PROJECT_LOG.md:38-61`).

That reveals an important lesson:

> authoritative legal PDF ingestion gave the project a more reliable source base than trying to treat a dynamic legal site as the main knowledge pipeline.

### Pivot 3: from section-only thinking to chunk + metadata + expansion thinking

The January 4 and January 7 entries describe a shift toward fixed-length chunking, metadata-rich indexing, asymmetric prompting, and section expansion (`PROJECT_LOG.md:175-188`, `PROJECT_LOG.md:194-212`).

This is one of the biggest conceptual improvements in the project.

The mistake it reveals is:

> legal text is too structurally rich to treat “whole section” as the universal retrieval unit, but generation still often needs section-level context.

The current code solves that with chunk retrieval + section expansion.

### Pivot 4: from unstructured parsing alone to runtime-hardening and GPU engineering

Later log entries show a shift from parsing logic improvements toward GPU provider fixes, ONNX Runtime/TensorRT optimization, DLL/shared-library handling, and throughput tuning (`PROJECT_LOG.md:257-305`, `PROJECT_LOG.md:319-343`).

This reveals another real lesson:

> once corpus scale grows, runtime engineering becomes part of product engineering.

In other words, the “hard part” stopped being just legal parsing quality and became end-to-end operational reliability.

---

## 18. Important dependencies and why they exist

The backend requirements file is large, but only some dependencies matter for the main architecture.

### Core application dependencies

- `fastapi` / `uvicorn`: backend web API (`backend/requirements.txt:129`, `backend/requirements.txt:798`)
- `langchain-openai`: DeepSeek-compatible chat client integration (`backend/requirements.txt:267-274`)
- `pinecone` + `langchain-pinecone`: vector store layer (`backend/requirements.txt:273`, `backend/requirements.txt:453-460`)

### Model/runtime dependencies

- `onnx`, `onnxruntime`, `onnxruntime-gpu` (`backend/requirements.txt:376-386`)
- `tensorrt`, `tensorrt-cu13*` (`backend/requirements.txt:661-674`)
- `optimum`, `optimum-onnx` (`backend/requirements.txt:395-400`)
- `transformers`, `torch` (`backend/requirements.txt:693-740`)

These exist because the repo is deeply invested in exported ONNX models and provider-controlled inference rather than plain eager-mode model execution.

### Parsing/document dependencies

- `unstructured`, `unstructured-inference`, `pdf2image`, `pdfplumber`, `pymupdf`, `pikepdf`, `google-cloud-vision`, and related OCR/layout packages (`backend/requirements.txt:430-787`)

These exist because legal PDF ingestion is structurally messy and needed robust layout/text extraction options.

### Frontend dependencies

- `react`, `react-dom`
- `react-markdown`, `remark-gfm`
- `lucide-react`
- Tailwind-related packages

(`frontend/package.json:12-33`)

These exist to support a lightweight, readable legal chat UI with markdown answers and reference cards.

---

## 19. Testing and verification map

The repo's testing approach is more script-driven than framework-heavy.

### Frontend verification

From AGENTS guidance, frontend changes should run:

- `npm run lint`
- optionally `npm run build`

(`AGENTS.md:90-93`)

### Backend runtime verification

The AGENTS guide explicitly lists the important backend commands:

- `python backend\main.py`
- `python backend\llm_evaluate.py`
- `python backend\scripts\ingest_pdfs.py --cap 282 599A`
- `python backend\scripts\verify_cpu_rerank_pipeline.py --query "<query>" --mode fast`
- `python backend\scripts\ollama_discover.py`
- `python backend\scripts\ollama_lifecycle.py status`
- `python backend\tests\test_dll.py`
- `python backend\tests\test_embedding_similarity.py`
- `python backend\tests\test_tensorrt_embedding.py`
- `python backend\tests\test_reranker_tokenizer_unicode.py`

(`AGENTS.md:49-60`)

### What each script proves

#### `backend/tests/test_dll.py`
Checks whether key DLL paths and CUDA-related dependencies can be loaded in Windows (`backend/tests/test_dll.py:40-80`).

#### `backend/tests/test_tensorrt_embedding.py`
Checks that:

- TensorRT provider is active
- CUDA EP is not active in embedding mode
- produced embeddings are non-zero

(`backend/tests/test_tensorrt_embedding.py:22-79`)

#### `backend/tests/test_reranker_tokenizer_unicode.py`
Checks that reranking still works when the query contains Unicode oddities and invisible characters (`backend/tests/test_reranker_tokenizer_unicode.py:30-64`).

#### `backend/scripts/verify_cpu_rerank_pipeline.py`
Verifies the retrieval/expansion/regroup path and can test reranker behavior in CPU mode (`backend/scripts/verify_cpu_rerank_pipeline.py:94-270`).

This is important because it lets the project isolate retrieval correctness from TensorRT-specific reranker instability.

---

## 20. Common invariants and failure traps

These are the things a future maintainer is most likely to break.

### Invariant 1: call `setup_cuda_dlls()` before importing torch or onnxruntime

This rule is stated both in docs and code:

- `AGENTS.md:66-68`
- `backend/core/setup_env.py:100-104`

You can see the code obeying this pattern in:

- `backend/main.py:20-30`
- `backend/services/vector_store.py:3-12`
- `backend/services/embedding_service.py:14-16`, then torch/onnx imports later
- `backend/services/reranker_service.py:17-23`
- `backend/scripts/ingest_pdfs.py:16-31`

Break this, and provider availability or library loading may fail in ways that look unrelated to the real cause.

### Invariant 2: do not mix Windows and WSL virtualenvs

This is documented repeatedly because it clearly burned the project before:

- `AGENTS.md:70-72`
- `backend/README.md:18-18`
- `backend/README.md:62-66`
- `backend/llm_evaluate.py:1327-1331`

### Invariant 3: SSE payload shape is strict

The frontend only parses lines starting with `data: ` and expects `answer`, `references`, or `error` payload keys (`frontend/src/components/ChatInterface.jsx:101-145`, `AGENTS.md:74-80`).

### Invariant 4: `session_id="default"` is invalid in live chat

That is enforced at the backend stream level (`backend/main.py:903-914`).

### Invariant 5: root-level backend duplicates are not the active path

The AGENTS guide explicitly warns about stale or non-primary duplicates (`AGENTS.md:18`).

### Invariant 6: `verify_trt.py` is stale

This is called out directly in the top-level AGENTS guide (`AGENTS.md:87-88`).

---

## 21. Suggested study order for a human reader

If you are starting from zero, read files in this order:

1. `AGENTS.md`
2. `backend/AGENTS.md`
3. `frontend/AGENTS.md`
4. `frontend/src/main.jsx`
5. `frontend/src/App.jsx`
6. `frontend/src/components/ChatInterface.jsx`
7. `backend/main.py`
8. `backend/services/vector_store.py`
9. `backend/core/utils.py`
10. `backend/services/embedding_service.py`
11. `backend/services/reranker_service.py`
12. `backend/scripts/ingest_pdfs.py`
13. `backend/parsers/pdf_parser.py`
14. `backend/core/setup_env.py`
15. `backend/core/ollama_runtime.py`
16. `PROJECT_LOG.md`
17. `docs/RAG_DESIGN.md`
18. `docs/PDF_PARSER_LOGIC.md`

That order gives you:

- contract first
- UI second
- live server third
- retrieval core fourth
- runtime/inference internals fifth
- ingestion and parsing sixth
- history and design docs last

---

## 22. Direct answers to your questions

### Q1. “We encountered errors that made the PDF ingestion script unable to run. At last we implemented worker and queue architecture. How did we achieve it and why did we do it?”

You achieved it by centralizing embedding work around:

- a shared queue `job_q` (`backend/core/embedding_shared.py:3-7`)
- a long-lived background worker `embedding_worker()` (`backend/scripts/ingest_pdfs.py:40-170`)
- per-request reply queues (`backend/scripts/ingest_pdfs.py:145-161`, `backend/parsers/pdf_parser.py:64-83`)
- a singleton `EmbeddingService` (`backend/services/embedding_service.py:39-61`)

You did it because embedding was not ordinary stateless function work. It was expensive, GPU-bound, TensorRT-sensitive runtime work. Reusing one shared runtime and routing requests through a worker made model lifecycle, batching, retries, cache behavior, and failure handling much more controllable.

So yes: your intuition about reusing one loaded session is basically correct, but the deeper reason is “one controlled inference runtime is safer and cheaper than many competing runtimes.”

### Q2. “What is TensorRT? What is the difference between TensorRT and CUDA?”

In the context of this repo:

- **CUDA** is the lower-level GPU compute/runtime foundation.
- **TensorRT** is a higher-level inference optimization/execution engine built for fast neural network inference on NVIDIA GPUs.
- **ONNX Runtime** is the host runtime that loads ONNX models and delegates execution to providers such as TensorRT or CPU.

So CUDA is not a substitute for TensorRT. TensorRT usually sits above the GPU runtime layer and specializes inference execution.

### Q3. “Why do we use CUDA and why does it cause accumulation in VRAM while TensorRT won't?”

Use more careful wording here:

- the project uses GPU-backed runtimes because embedding and reranking throughput matter
- apparent VRAM “accumulation” is often allocator or cache behavior rather than a literal leak
- TensorRT and generic CUDA-backed execution can show different memory patterns, but both can retain GPU memory depending on runtime behavior

So the accurate answer is not “TensorRT won't accumulate VRAM.” It is “the memory behavior can look different because the runtimes manage memory differently, and this project explicitly uses cache directories, warmup, and unload logic to control that behavior.”

### Q4. “What is ONNX Runtime and why do we need TensorRT and ONNX Runtime?”

ONNX Runtime is the model execution framework. TensorRT is one possible accelerated backend under ONNX Runtime.

You need ONNX Runtime because the project relies on exported ONNX models and wants a common interface for running them. You need TensorRT because the project wants a faster, optimized provider path for heavy inference workloads, especially embeddings during ingestion.

### Q5. “How do we setup WSL for this project?”

The repo-supported path is:

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Then, if TensorRT/ONNX Runtime GPU behavior is needed in WSL, use the repo's supporting setup scripts/notes:

- `backend/scripts/setup_wsl_libs.sh`
- `backend/scripts/run_trt_wsl.sh`
- `.sisyphus/drafts/wsl-setup-instructions.md`

And if Ollama lives on the Windows host, use `backend/scripts/ollama_discover.py` or the README guidance to discover a host-reachable `OLLAMA_BASE_URL`.

### Q6. “Why do we use WSL to run scripts involving embedding service or reranking? Why does WSL make them stable but not Windows?”

The best repo-grounded answer is:

- the project uses separate Windows and WSL environments because GPU inference dependencies are platform-sensitive
- WSL gives the project a Linux-style runtime where TensorRT shared-library loading and ONNX Runtime GPU integration can be made more predictable
- the repo contains specific WSL support code and WSL networking logic, which means WSL became part of the intended stable workflow
- native Windows is still supported for many tasks, but the project learned to avoid mixed environments and to use WSL when Linux-style GPU library behavior was more dependable for embedding/reranking workflows

So WSL did not solve the business logic. It solved part of the runtime-environment instability.

---

## 23. Glossary

- **SSE**: Server-Sent Events. A streaming HTTP response format where the server pushes incremental `data:` messages.
- **Embedding**: Numeric vector representation of text used for similarity search.
- **Reranker**: Model that re-scores retrieved candidates more precisely than raw vector similarity.
- **HyDE**: Hypothetical Document Embeddings. Generate a plausible legal-text-like passage first, then embed that to improve retrieval.
- **Section expansion**: Retrieve a chunk, then reconstruct the whole legal section by fetching sibling chunks.
- **TensorRT EP**: TensorRT execution provider under ONNX Runtime.
- **CUDA EP**: CUDA execution provider under ONNX Runtime.
- **Singleton runtime**: One process-wide service instance reused across callers.
- **Reply queue**: Per-job queue used to return results from the worker to the original requester.
- **Precision regime**: The embedding precision assumptions (for example FP16 vs FP32) under which vectors were produced.

---

## 24. Source index

### Repo contracts

- `AGENTS.md:10-18`
- `AGENTS.md:18`
- `AGENTS.md:66-80`
- `AGENTS.md:87-93`

### Frontend

- `frontend/src/main.jsx:1-10`
- `frontend/src/App.jsx:4-23`
- `frontend/src/components/ChatInterface.jsx:7-18`
- `frontend/src/components/ChatInterface.jsx:20-31`
- `frontend/src/components/ChatInterface.jsx:48-160`
- `frontend/src/components/ChatInterface.jsx:162-281`
- `frontend/src/components/ReferenceCard.jsx:4-40`
- `frontend/src/index.css:1-22`
- `frontend/package.json:6-35`
- `frontend/vite.config.js:1-7`
- `frontend/eslint.config.js:1-29`

### Backend chat and memory

- `backend/main.py:199-205`
- `backend/main.py:208-223`
- `backend/main.py:435-439`
- `backend/main.py:898-1052`

### Retrieval

- `backend/core/utils.py:9-243`
- `backend/services/vector_store.py:22-228`
- `backend/services/vector_store.py:229-533`
- `backend/services/reranker_service.py:188-365`
- `backend/services/reranker_service.py:520-1006`

### Embedding runtime

- `backend/services/embedding_service.py:39-121`
- `backend/services/embedding_service.py:142-157`
- `backend/services/embedding_service.py:159-284`
- `backend/services/embedding_service.py:286-429`

### Ingestion and parser

- `backend/scripts/ingest_pdfs.py:16-34`
- `backend/scripts/ingest_pdfs.py:40-170`
- `backend/scripts/ingest_pdfs.py:182-198`
- `backend/scripts/ingest_pdfs.py:216-270`
- `backend/scripts/ingest_pdfs.py:276-611`
- `backend/parsers/pdf_parser.py:61-116`
- `backend/parsers/pdf_parser.py:121-199`
- `backend/parsers/pdf_parser.py:202-455`
- `backend/core/embedding_shared.py:3-7`

### Runtime/platform setup

- `backend/core/setup_env.py:7-176`
- `backend/core/ollama_runtime.py:162-344`
- `backend/README.md:18-30`
- `backend/README.md:62-103`
- `backend/requirements.txt:693-726`

### Verification

- `backend/tests/test_dll.py:40-80`
- `backend/tests/test_tensorrt_embedding.py:17-84`
- `backend/tests/test_reranker_tokenizer_unicode.py:30-68`
- `backend/scripts/verify_cpu_rerank_pipeline.py:94-274`

### Historical and secondary design context

- `PROJECT_LOG.md:37-76`
- `PROJECT_LOG.md:143-188`
- `PROJECT_LOG.md:194-212`
- `PROJECT_LOG.md:257-343`
- `docs/RAG_DESIGN.md:39-128`
- `docs/PDF_PARSER_LOGIC.md:7-58`
- `.sisyphus/plans/fix-embedding-deadlock.md:20-36`
- `.sisyphus/plans/fix-tensorrt-cuda-runtime-error.md:20-83`
- `.sisyphus/plans/yuan-onnx-trt-wsl.md:20-66`

---

## Final takeaway

If you only remember one thing from this entire repo, remember this:

> The project is not just a chatbot. It is a legal indexing-and-retrieval system with a chatbot interface on top. The hardest engineering problems were not only prompt design or frontend UX. They were corpus construction, retrieval quality, and stable shared GPU inference under real operational constraints.

That is why the repo now looks the way it does.
