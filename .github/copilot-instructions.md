# Copilot Instructions

## Commands

- **Local preview:** `python3 -m http.server 8000`
- Open `http://localhost:8000/` from the repository root.
- Use a local HTTP server instead of opening files directly, because note pages load Markdown with `fetch(...)`.
- There are no project-defined build, test, lint, or single-test commands in this repository.

## High-level architecture

- This repository is a static personal site served directly from the repo root. There is no app framework, package manager, or build pipeline.
- Root entry pages are hand-authored HTML files such as `index.html`, `photos.html`, `CourseNotes.html`, `ReadingPapers.html`, and `Music.html`.
- Shared site-wide presentation lives in `assets/css/style.css`, and shared interaction/animation lives in `assets/js/script.js`.
- `Photos/*.html` contains individual photography collection pages. Images are stored under `assets/images/...`, and album/list pages link to those HTML pages directly.
- `CourseNotes/*.html`, `CourseNotes/*/*.html`, and `ReadingPaper/*.html` follow a two-level structure:
  - list/index pages are plain HTML; top-level `CourseNotes.html` and `ReadingPapers.html` now render cards from `assets/data/content-manifest.json`
  - note/detail pages are HTML shells that fetch sibling Markdown from `Markdown/*.md`, then render it client-side with `marked`, `highlight.js`, and `KaTeX` through `assets/js/note-renderer.js`
- `assets/data/content-manifest.json` is the minimal content manifest for top-level note/paper list pages; update this file when adding new courses or reading-paper entries.
- The music page is self-contained in `Music.html`: UI markup is HTML/CSS, and playlist metadata is an inline JavaScript array rather than a separate data file.

## Key conventions

- Keep the shared cyberpunk/neon look in `assets/css/style.css`; page-specific layout changes are usually implemented as inline `<style>` blocks inside the page being edited.
- Most pages include `<div class="background-glow"></div>` plus the shared `assets/js/script.js` background animation. Preserve those when creating new pages in the same style.
- Relative asset paths depend on directory depth:
  - root pages use `assets/...`
  - one-level nested pages use `../assets/...`
  - two-level nested note pages use `../../assets/...`
- Navigation is manual. When adding or renaming content, update the relevant parent listing page as well as the destination page; there is no generated routing or shared content registry.
- For Markdown-backed note pages, keep the existing shell pattern:
  - set `window.__NOTE_CONFIG__ = { markdownFile: './Markdown/<name>.md' }`
  - load `assets/js/note-renderer.js` (relative path depends on nesting level)
  - keep `assets/js/script.js` for global background/lightbox behavior
  - preserve `marked`, `highlight.js`, and `KaTeX` CDN includes (the shared renderer depends on them)
- Image layout inside Markdown is controlled through words in the image alt text. The note shells map `right`, `left`, `small`, `medium`, and `large` to CSS classes after rendering.
- The photo lightbox code in `assets/js/script.js` only activates on pages that provide the expected DOM hooks (`#lightbox`, `#lightbox-img`, `#caption`, `.close-btn`, and `.gallery-container`). Stream-style photo pages without that markup only use the shared background animation.
