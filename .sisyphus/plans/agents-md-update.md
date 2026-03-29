# Tighten Root AGENTS Guide for Coding Agents

## TL;DR
> **Summary**: Refresh the repository root `AGENTS.md` so it stays repo-specific, stays near the requested ~150-line target, and gives coding agents exact command and style guidance grounded in the current backend/frontend setup.
> **Deliverables**:
> - Updated root `AGENTS.md`
> - Explicit backend single-test workflow documentation
> - Explicit AGENTS precedence and editor-rule absence note
> - Executable verification evidence for content and line count
> **Effort**: Short
> **Parallel**: NO
> **Critical Path**: 1 → 2 → 3 → 4

## Context
### Original Request
Create or improve `AGENTS.md` at `C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot` so agentic coding agents receive build/lint/test commands, especially single-test commands, plus code-style guidance around imports, formatting, typing, naming, and error handling. Keep the file about 150 lines and incorporate Cursor/Copilot rules if present.

### Interview Summary
- Existing root `AGENTS.md` already covers most required areas but is 172 lines and can be tightened.
- Backend tests are standalone Python scripts under `backend/tests/`; there is no repo-level pytest workflow.
- Frontend commands are sourced from `frontend/package.json` and should remain the canonical build/lint workflow.
- No `.cursor/rules/**`, `.cursorrules`, or `.github/copilot-instructions.md` files were found.

### Metis Review (gaps addressed)
- Preserve repo-specific constraints while compressing wording rather than replacing with generic AI-policy prose.
- Add an explicit precedence statement so agents know root guidance is overridden by deeper `backend/AGENTS.md` and `frontend/AGENTS.md` within those subtrees.
- Avoid speculative commands, CI references, or generic `pytest` instructions.
- Verify the final file with executable content assertions and an exact line-count check.

## Work Objectives
### Core Objective
Produce a tighter, agent-facing root `AGENTS.md` that accurately reflects current repo workflows and conventions without editing any source files or subdirectory AGENTS files.

### Deliverables
- A revised root `AGENTS.md` at repository root
- Exact backend run/install/single-test commands
- Exact frontend install/dev/build/lint/preview commands
- Code-style guidance for Python and JavaScript/JSX grounded in current repo conventions
- A short note confirming no repo-level Cursor/Copilot rule files exist
- Verification artifacts under `.sisyphus/evidence/`

### Definition of Done (verifiable conditions with commands)
- Root `AGENTS.md` exists and remains near the requested target:
  - `python -c "from pathlib import Path; print(len(Path('AGENTS.md').read_text(encoding='utf-8').splitlines()))"`
- Root `AGENTS.md` documents backend single-test commands exactly:
  - `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8'); assert 'python backend/tests/test_dll.py' in t and 'python backend/tests/test_embedding_similarity.py' in t and 'python backend/tests/test_tensorrt_embedding.py' in t; print('ok')"`
- Root `AGENTS.md` documents canonical frontend commands:
  - `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8'); assert all(x in t for x in ['npm install','npm run dev','npm run build','npm run lint','npm run preview']); print('ok')"`
- Root `AGENTS.md` includes AGENTS hierarchy/editor-rule guidance:
  - `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8').lower(); assert 'backend/agents.md' in t and 'frontend/agents.md' in t and ('cursor' in t or 'copilot' in t); print('ok')"`

### Must Have
- Improve the existing root file instead of replacing repo-specific guidance with generic text
- Keep scope limited to root `AGENTS.md`
- State that backend tests are standalone scripts, not pytest
- Include exact frontend commands from `frontend/package.json`
- Include explicit import, formatting, naming, typing, styling, and error-handling guidance
- Add an explicit precedence sentence for root vs subdirectory `AGENTS.md`

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No edits to `backend/AGENTS.md` or `frontend/AGENTS.md`
- No generic pytest/full-suite instructions unsupported by the repo
- No invented CI, formatter, or build commands
- No speculative editor-rule content when no such files exist
- No broad “best practices” prose detached from actual repository files

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after using Python content assertions and file-existence checks
- QA policy: Every task includes agent-executed happy-path and failure-path checks
- Evidence: `.sisyphus/evidence/task-{N}-agents-md-update.{ext}`

## Execution Strategy
### Parallel Execution Waves
> This task is intentionally sequential because each later step depends on the edited root document from the prior step.

Wave 1: Task 1 — command section + hierarchy statement

Wave 2: Task 2 — style guidance + anti-patterns + editor-rule note

Wave 3: Task 3 — compression and delegation to subdirectory AGENTS

Wave 4: Task 4 — executable verification and final polish

### Dependency Matrix (full, all tasks)
| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | None | 2, 3, 4 |
| 2 | 1 | 3, 4 |
| 3 | 2 | 4 |
| 4 | 3 | F1-F4 |

### Agent Dispatch Summary (wave → task count → categories)
| Wave | Task Count | Category |
|------|------------|----------|
| 1 | 1 | writing |
| 2 | 1 | writing |
| 3 | 1 | writing |
| 4 | 1 | quick |

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Refresh command section and AGENTS hierarchy note

  **What to do**: Edit the root `AGENTS.md` in place. Preserve the existing project overview/structure at a compressed level, then make the command section explicitly agent-facing. Keep backend commands rooted at repository root and frontend commands rooted at `frontend/`. Add a short hierarchy statement near the top stating that `backend/AGENTS.md` and `frontend/AGENTS.md` override root guidance for files in those subtrees.
  **Must NOT do**: Do not edit `backend/AGENTS.md` or `frontend/AGENTS.md`. Do not add `pytest`, `ruff check`, `npm test`, or CI commands not already supported by the repo.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: This is a repo-guidance documentation rewrite with precision requirements.
  - Skills: `[]` — No extra skill required.
  - Omitted: `['git-master']` — No git operation is required for the task itself.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2, 3, 4] | Blocked By: []

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:36-68` — Existing command structure to tighten rather than replace.
  - Pattern: `AGENTS.md:34` — Existing mention of subdirectory AGENTS files that should become an explicit precedence rule.
  - API/Type: `frontend/package.json:6-10` — Canonical frontend scripts.
  - Test: `backend/tests/test_dll.py:39-40` — Standalone script execution via `__main__`.
  - Test: `backend/tests/test_embedding_similarity.py:99-100` — Standalone script execution via `main()`.
  - Test: `backend/tests/test_tensorrt_embedding.py:75-76` — Standalone script execution with exit code.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `AGENTS.md` contains the exact strings `python backend/main.py`, `python backend/tests/test_dll.py`, `python backend/tests/test_embedding_similarity.py`, and `python backend/tests/test_tensorrt_embedding.py`.
  - [ ] `AGENTS.md` contains the exact strings `npm install`, `npm run dev`, `npm run build`, `npm run lint`, and `npm run preview`.
  - [ ] `AGENTS.md` contains an explicit sentence that root guidance is overridden by `backend/AGENTS.md` and `frontend/AGENTS.md` within those directories.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Happy path command coverage
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8'); assert all(x in t for x in ['python backend/main.py','python backend/tests/test_dll.py','python backend/tests/test_embedding_similarity.py','python backend/tests/test_tensorrt_embedding.py','npm install','npm run dev','npm run build','npm run lint','npm run preview']); print('ok')"` from repo root and save stdout to `.sisyphus/evidence/task-1-agents-md-update.txt`.
    Expected: Command prints `ok`.
    Evidence: .sisyphus/evidence/task-1-agents-md-update.txt

  Scenario: Failure path unsupported commands
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8').lower(); forbidden=['pytest','npm test','ruff check']; assert not any(x in t for x in forbidden); print('ok')"` and save stdout to `.sisyphus/evidence/task-1-agents-md-update-error.txt`.
    Expected: Command prints `ok`; if it fails, remove unsupported commands before proceeding.
    Evidence: .sisyphus/evidence/task-1-agents-md-update-error.txt
  ```

  **Commit**: NO | Message: `docs: tighten root AGENTS guide for coding agents` | Files: [`AGENTS.md`]

- [x] 2. Tighten backend/frontend style guidance without losing repo-specific constraints

  **What to do**: Rewrite the style sections so they stay concise but retain concrete rules already validated from repo files. Backend must cover imports, formatting, naming, typing, error handling, `setup_env.setup_cuda_dlls()` ordering, `.env` handling, and direct-script patterns. Frontend must cover ESLint 9 flat config, two-space indentation, single quotes, `no-unused-vars` behavior, functional components, Tailwind-only styling, class-based dark mode, and local state/fetch/SSE patterns.
  **Must NOT do**: Do not add TypeScript guidance, Redux guidance as a requirement, structured logging mandates, or formatter rules not present in the repo.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: Condensing technical conventions while preserving precision is a documentation task.
  - Skills: `[]` — No extra skill required.
  - Omitted: `['frontend-ui-ux']` — This is guidance writing, not interface design.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [3, 4] | Blocked By: [1]

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:70-138` — Existing style guidance to tighten.
  - API/Type: `frontend/eslint.config.js:7-29` — ESLint 9 flat config, `dist` ignore, JSX files, `no-unused-vars` rule.
  - API/Type: `frontend/tailwind.config.js:2-11` — `darkMode: 'class'` and frontend content globs.
  - Pattern: `backend/AGENTS.md:32-41` — Backend local conventions and anti-pattern reminders.
  - Pattern: `frontend/AGENTS.md:28-39` — Frontend local conventions and anti-pattern reminders.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `AGENTS.md` mentions backend absolute `backend.*` imports and notes `main.py` bare imports are an exception.
  - [ ] `AGENTS.md` mentions `setup_env.setup_cuda_dlls()` must run before torch/onnxruntime imports.
  - [ ] `AGENTS.md` mentions ESLint 9 flat config and the `no-unused-vars` ignore pattern for names matching `^[A-Z_]`.
  - [ ] `AGENTS.md` mentions Tailwind utility classes and class-based dark mode.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Happy path style coverage
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8'); checks=['backend.*','setup_env.setup_cuda_dlls()','ESLint 9','no-unused-vars','Tailwind','dark mode']; assert all(x in t for x in checks); print('ok')"` and save stdout to `.sisyphus/evidence/task-2-agents-md-update.txt`.
    Expected: Command prints `ok`.
    Evidence: .sisyphus/evidence/task-2-agents-md-update.txt

  Scenario: Failure path generic drift
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8').lower(); forbidden=['typescript required','redux required','prettier']; assert not any(x in t for x in forbidden); print('ok')"` and save stdout to `.sisyphus/evidence/task-2-agents-md-update-error.txt`.
    Expected: Command prints `ok`; if it fails, replace generic or unsupported guidance with repo-backed wording.
    Evidence: .sisyphus/evidence/task-2-agents-md-update-error.txt
  ```

  **Commit**: NO | Message: `docs: tighten root AGENTS guide for coding agents` | Files: [`AGENTS.md`]

- [x] 3. Compress the root file to target size and add editor-rule absence guidance

  **What to do**: Reduce redundancy in the root `AGENTS.md` so the final file lands near the user’s requested ~150-line target while preserving operationally important content. Prefer shortening headings, merging overlapping bullets, and delegating deeper details to `backend/AGENTS.md` and `frontend/AGENTS.md`. Add a concise note that no `.cursor/rules/**`, `.cursorrules`, or `.github/copilot-instructions.md` files were found, so agents should follow AGENTS files only.
  **Must NOT do**: Do not remove the backend single-test examples, frontend command list, hierarchy note, or critical CUDA/env/Tailwind constraints just to hit a number.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: This is a precision editing and compression step.
  - Skills: `[]` — No extra skill required.
  - Omitted: `['oracle']` — Formal review belongs to the final verification wave, not the implementation step.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: [4] | Blocked By: [2]

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:1-172` — Current root file length and sections to compress.
  - Pattern: `backend/AGENTS.md:20-45` — Backend-specific detail that can be referenced rather than duplicated.
  - Pattern: `frontend/AGENTS.md:18-48` — Frontend-specific detail that can be referenced rather than duplicated.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Final `AGENTS.md` line count is between 140 and 170 inclusive unless preserving a critical repo-specific constraint requires one-digit overflow.
  - [ ] `AGENTS.md` explicitly states no repo-level Cursor/Copilot instruction files were found.
  - [ ] `AGENTS.md` still directs agents to `backend/AGENTS.md` and `frontend/AGENTS.md` for subtree-specific guidance.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Happy path target size
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; print(len(Path('AGENTS.md').read_text(encoding='utf-8').splitlines()))"` and save stdout to `.sisyphus/evidence/task-3-agents-md-update.txt`.
    Expected: Printed line count is between 140 and 170 inclusive.
    Evidence: .sisyphus/evidence/task-3-agents-md-update.txt

  Scenario: Failure path missing editor-rule note
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8').lower(); assert 'cursor' in t and 'copilot' in t; print('ok')"` and save stdout to `.sisyphus/evidence/task-3-agents-md-update-error.txt`.
    Expected: Command prints `ok`; if it fails, add the absence note before proceeding.
    Evidence: .sisyphus/evidence/task-3-agents-md-update-error.txt
  ```

  **Commit**: NO | Message: `docs: tighten root AGENTS guide for coding agents` | Files: [`AGENTS.md`]

- [x] 4. Run executable validation against file content and referenced scripts

  **What to do**: After the root file is updated, run file-existence and content-assertion checks from repo root and collect evidence in `.sisyphus/evidence/`. Confirm every documented single-test file still exists, every documented frontend command is still present in `frontend/package.json`, and the root file contains the required hierarchy/style/editor-note content.
  **Must NOT do**: Do not stop at visual inspection. Do not claim success if any check fails; fix `AGENTS.md` first, then rerun all checks.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: This is a bounded verification step with exact commands.
  - Skills: `[]` — No extra skill required.
  - Omitted: `['playwright']` — No browser interaction is needed.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: [F1, F2, F3, F4] | Blocked By: [3]

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `frontend/package.json:6-10` — Canonical frontend commands to verify.
  - Test: `backend/tests/test_dll.py` — File existence verification.
  - Test: `backend/tests/test_embedding_similarity.py` — File existence verification.
  - Test: `backend/tests/test_tensorrt_embedding.py` — File existence verification.
  - Pattern: `.sisyphus/plans/agents-md-update.md:39-44` — Definition-of-done commands to execute.

  **Acceptance Criteria** (agent-executable only):
  - [ ] File existence check passes for all three documented backend test scripts.
  - [ ] Content assertion passes for frontend commands in `AGENTS.md`.
  - [ ] Content assertion passes for hierarchy/editor-note content in `AGENTS.md`.
  - [ ] Evidence files are created under `.sisyphus/evidence/` for all verification commands.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Happy path complete validation
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; files=['backend/tests/test_dll.py','backend/tests/test_embedding_similarity.py','backend/tests/test_tensorrt_embedding.py']; assert all(Path(f).exists() for f in files); t=Path('AGENTS.md').read_text(encoding='utf-8'); assert all(x in t for x in ['npm install','npm run dev','npm run build','npm run lint','npm run preview']); print('ok')"` and save stdout to `.sisyphus/evidence/task-4-agents-md-update.txt`.
    Expected: Command prints `ok`.
    Evidence: .sisyphus/evidence/task-4-agents-md-update.txt

  Scenario: Failure path hierarchy/content regression
    Tool: Bash
    Steps: Run `python -c "from pathlib import Path; t=Path('AGENTS.md').read_text(encoding='utf-8').lower(); assert 'backend/agents.md' in t and 'frontend/agents.md' in t and 'cursor' in t and 'copilot' in t; print('ok')"` and save stdout to `.sisyphus/evidence/task-4-agents-md-update-error.txt`.
    Expected: Command prints `ok`; if it fails, restore the missing guidance and rerun all validation commands.
    Evidence: .sisyphus/evidence/task-4-agents-md-update-error.txt
  ```

  **Commit**: YES | Message: `docs: tighten root AGENTS guide for coding agents` | Files: [`AGENTS.md`]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- One atomic docs-only commit after verification passes
- Recommended message: `docs: tighten root AGENTS guide for coding agents`
- Files: `AGENTS.md`

## Success Criteria
- Root `AGENTS.md` is more concise than the current 172-line version while still preserving validated repo-specific instructions
- Agents can discover exact backend single-test commands without guessing
- Agents can discover exact frontend build/lint/dev commands without opening other files
- Agents can see the repo’s Python/JS conventions and major anti-patterns in one place
- Agents are told that subdirectory `AGENTS.md` files carry more specific subtree guidance
- No unsupported Cursor/Copilot rules are invented
