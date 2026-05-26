# Yuchen Wu's Personal Website

A beautifully designed static personal website showcasing course notes, paper reviews, photography, music, and more. Built with vanilla HTML, CSS, and JavaScript with a cyberpunk/neon aesthetic.

## 🌐 Live Demo

Visit the live site: [https://AaronWu77.github.io](https://AaronWu77.github.io)

## 📚 About

This is the personal portfolio website of Yuchen Wu, a junior student at Zhejiang University majoring in Computer Science and Technology, with research interests in deep learning and computer vision.

The site showcases:
- **📖 Course Notes** - Study notes from university courses (Linear Algebra, Physics, Software Engineering, Data Structures, Computer Architecture, Digital Logic, Compilers, IELTS)
- **📰 Reading Papers** - Reviews and summaries of academic papers (primarily computer vision and deep learning)
- **📸 Photography** - Personal photography collections and gallery
- **🎵 Music Playlist** - Personal music playlist
- **📄 Publications** - Links to published research papers

## 🏗️ Project Structure

```
.
├── index.html                          # Homepage
├── CourseNotes.html                    # Course notes list page
├── ReadingPapers.html                  # Reading papers list page
├── Music.html                          # Music playlist page
├── photos.html                         # Photography gallery list
│
├── CourseNotes/                        # Individual course note pages
│   ├── 线性代数.html
│   ├── 普通物理学.html
│   ├── IELTS.html
│   └── ...
│
├── ReadingPaper/                       # Individual paper review pages
│   ├── paper1.html
│   ├── paper2.html
│   └── ...
│
├── Photos/                             # Photography collection pages
│   └── *.html
│
├── Markdown/                           # Markdown content files
│   └── *.md                           # Rendered by note-renderer.js
│
├── assets/
│   ├── css/
│   │   └── style.css                  # Main stylesheet (cyberpunk theme)
│   ├── js/
│   │   ├── script.js                  # Global animations and interactions
│   │   └── note-renderer.js           # Markdown rendering for note pages
│   ├── images/                        # Image assets
│   └── data/
│       └── content-manifest.json      # Content index for list pages
│
└── doc/                               # Documentation
    └── plan.md                        # Development plan
```

## 💻 Technology Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Markdown Rendering**: [marked.js](https://marked.js.org/) for Markdown parsing
- **Code Highlighting**: [Highlight.js](https://highlightjs.org/) for syntax highlighting
- **Math Rendering**: [KaTeX](https://katex.org/) for LaTeX equations
- **Hosting**: GitHub Pages

## 🎨 Design

The site features a distinctive **cyberpunk/neon aesthetic** with:
- Neon cyan (#00f3ff) and magenta highlights
- Interactive background animations
- Responsive design for all screen sizes
- Smooth transitions and hover effects
- Lightbox gallery for photography

## 🚀 Quick Start

### Local Preview

To preview the site locally, serve it with a local HTTP server:

```bash
cd /path/to/AaronWu77.github.io
python3 -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**Note**: Local file serving (`file://`) won't work for the Markdown content fetching. Use an HTTP server as shown above.

## 📝 Content Management

### Adding Course Notes

1. Create a new Markdown file in `Markdown/` (e.g., `MyNewCourse.md`)
2. Create a new HTML shell in `CourseNotes/MyNewCourse.html`:
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <meta charset="UTF-8">
       <title>My New Course - Yuchen Wu</title>
       <link rel="stylesheet" href="../../assets/css/style.css">
       <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.css">
       <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.css">
   </head>
   <body>
       <div class="background-glow"></div>
       <div id="content"></div>
       <script>window.__NOTE_CONFIG__ = { markdownFile: '../../Markdown/MyNewCourse.md' };</script>
       <script src="../../assets/js/note-renderer.js"></script>
       <script src="../../assets/js/script.js"></script>
   </body>
   </html>
   ```
3. Update `assets/data/content-manifest.json` to include the new course in the `courseNotes` array:
   ```json
   {
     "title": "My New Course",
     "date": "YYYY.M.D",
     "summary": "Description of the course",
     "url": "./CourseNotes/MyNewCourse.html"
   }
   ```

### Adding Paper Reviews

1. Create a new HTML file in `ReadingPaper/` (e.g., `paper3.html`)
2. Update `assets/data/content-manifest.json` to include it in the `readingPapers` array

### Adding Photos

1. Add images to `assets/images/`
2. Create a gallery HTML page in `Photos/`
3. Update `photos.html` with links to the new gallery

## 🎯 Key Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark Theme**: Easy on the eyes with cyberpunk aesthetics
- **Markdown Support**: Write course notes in Markdown with code highlighting and LaTeX
- **Lightbox Gallery**: Smooth image viewing experience
- **Fast Loading**: Static HTML, no backend required
- **SEO Ready**: Proper HTML structure and metadata

## 📋 Relative Path Convention

Asset paths depend on the directory depth:
- **Root pages** (e.g., `index.html`): use `assets/...`
- **One-level nested** (e.g., `CourseNotes/*.html`): use `../assets/...`
- **Two-level nested** (e.g., `CourseNotes/*/*.html` or `ReadingPaper/*.html`): use `../../assets/...`

## 🔧 Development Tips

1. **Global Styles**: Modify `assets/css/style.css` to change site-wide styling
2. **Page-Specific Styles**: Add inline `<style>` blocks within individual pages
3. **Global Interactions**: Update `assets/js/script.js` for site-wide JavaScript
4. **Note Rendering**: The `note-renderer.js` handles Markdown rendering with marked.js, KaTeX, and Highlight.js

## 📄 License

This project is a personal website and portfolio. The content and design are the intellectual property of Yuchen Wu.

## 📞 Contact

For inquiries, please reach out through the website or contact information provided on the homepage.

---

**Last Updated**: 2026-05-26