# FRONTEND KNOWLEDGE BASE

## OVERVIEW
React/Vite-based UI for interacting with the Hong Kong Legal Chatbot.

## STRUCTURE
```
frontend/
├── public/       # Static assets
└── src/
    ├── assets/     # Images and global styles
    ├── components/ # Reusable UI components
    ├── App.jsx     # Main application container and logic
    └── main.jsx    # Vite entry point
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| UI Layout | `src/App.jsx` | Main state for chat and document list |
| Styling | `src/index.css` | Global Tailwind directives |
| Components | `src/components/` | Modular UI elements (Chat, Sidebar) |
| Vite Config | `vite.config.js` | Build and dev server settings |

## CONVENTIONS
- **Styling**: Tailwind CSS 4.0 is used for all layout and component styling.
- **Framework**: React 19.2.0 (functional components with hooks).
- **ESM**: Strictly uses ES modules (`type: "module"` in `package.json`).

## ANTI-PATTERNS
- Avoid prop drilling; use context or state management if complexity grows.
- Do not add CSS files for individual components; prefer Tailwind utility classes.

## COMMANDS
```bash
npm install   # Install dependencies
npm run dev   # Start development server
npm run build # Build for production
```
