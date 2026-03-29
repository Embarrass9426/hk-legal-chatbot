# FRONTEND KNOWLEDGE BASE

## OVERVIEW
React/Vite-based UI for interacting with the Hong Kong Legal Chatbot.

## STRUCTURE
```
frontend/
├── public/       # Static assets
└── src/
    ├── assets/     # Images and global styles
    ├── components/ # ChatInterface.jsx, ReferenceCard.jsx
    ├── App.jsx     # Root component (dark mode state)
    ├── main.jsx    # Vite entry point
    └── index.css   # Tailwind directives + prose overrides
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Chat UI | `src/components/ChatInterface.jsx` | Streaming chat with SSE |
| Citations | `src/components/ReferenceCard.jsx` | Legal reference display |
| App Shell | `src/App.jsx` | Dark mode toggle, root state |
| Styling | `src/index.css` | Global Tailwind directives + prose |
| Vite Config | `vite.config.js` | Build and dev server settings |
| ESLint | `eslint.config.js` | ESLint 9 flat config |

## CONVENTIONS
- **Styling**: Tailwind CSS 4.0 utility classes exclusively. Dark mode via `dark:` variant.
- **Framework**: React 19.2.0 (functional components with hooks only).
- **ESM**: Strictly uses ES modules (`type: "module"` in `package.json`).
- **Icons**: `lucide-react`. **Markdown**: `react-markdown` + `remark-gfm`.
- **State**: Local `useState`/`useEffect`. No Redux/Context.
- **API**: Direct `fetch()` to `http://localhost:8000`. SSE streaming via `ReadableStream`.

## ANTI-PATTERNS
- Do not add CSS files for individual components — use Tailwind utility classes.
- Do not add npm dependencies if `lucide-react` or existing deps already cover the need.
- Avoid prop drilling; use context or state management if complexity grows.

## COMMANDS
```bash
npm install          # Install dependencies
npm run dev          # Vite dev server with HMR
npm run build        # Production build (vite build)
npm run lint         # ESLint check (eslint .)
npm run preview      # Preview production build
```
