(function () {
    function getMarkdownFromUrl() {
        const params = new URLSearchParams(window.location.search);
        return params.get('md');
    }

    function isSafeMarkdownPath(path) {
        if (!path || typeof path !== 'string') return false;
        const trimmed = path.trim();
        const lower = trimmed.toLowerCase();
        if (trimmed.includes('..')) return false;
        if (trimmed.startsWith('/') || trimmed.startsWith('//')) return false;
        if (/^[a-z][a-z0-9+.-]*:/.test(lower)) return false;
        return lower.endsWith('.md');
    }

    function renderError(outputEl, message) {
        outputEl.innerHTML = '<p style="color: red; text-align: center;">' + message + '</p>';
    }

    function renderMarkdownNote(config) {
        const outputId = (config && config.outputId) || 'markdown-output';
        const markdownFile = (config && config.markdownFile) || getMarkdownFromUrl();
        const outputEl = document.getElementById(outputId);

        if (!outputEl) {
            return;
        }

        if (!markdownFile) {
            renderError(outputEl, 'No markdown file was specified. Please provide a valid md query parameter or __NOTE_CONFIG__.');
            return;
        }

        if (!isSafeMarkdownPath(markdownFile)) {
            renderError(outputEl, 'Invalid markdown file path. Only relative .md paths are allowed.');
            return;
        }

        fetch(markdownFile)
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then((markdownText) => {
                const markdownDir = markdownFile.substring(0, markdownFile.lastIndexOf('/') + 1);
                markdownText = markdownText.replace(/!\[([^\]]*)\]\(((?!http|\/\/|\/)[^\)]+)\)/g, '![$1](' + markdownDir + '$2)');

                const mathPlaceholder = [];
                const protectMath = (str) => {
                    str = str.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
                        mathPlaceholder.push(match);
                        return '%%%MATH_BLOCK_' + (mathPlaceholder.length - 1) + '%%%';
                    });
                    str = str.replace(/\$([^\n\$]+?)\$/g, (match) => {
                        mathPlaceholder.push(match);
                        return '%%%MATH_INLINE_' + (mathPlaceholder.length - 1) + '%%%';
                    });
                    return str;
                };

                const protectedText = protectMath(markdownText);
                let htmlContent = marked.parse(protectedText);

                htmlContent = htmlContent.replace(/%%%MATH_(BLOCK|INLINE)_(\d+)%%%/g, (match, type, id) => mathPlaceholder[id]);
                outputEl.innerHTML = htmlContent;

                if (window.hljs) {
                    hljs.highlightAll();
                }

                document.querySelectorAll('#' + outputId + ' img').forEach((img) => {
                    const alt = img.alt.toLowerCase();
                    if (alt.includes('right')) img.classList.add('align-right');
                    if (alt.includes('left')) img.classList.add('align-left');
                    if (alt.includes('small')) img.classList.add('width-small');
                    if (alt.includes('medium')) img.classList.add('width-medium');
                    if (alt.includes('large')) img.classList.add('width-large');
                });

                if (window.renderMathInElement) {
                    renderMathInElement(outputEl, {
                        delimiters: [
                            { left: '$$', right: '$$', display: true },
                            { left: '$', right: '$', display: false },
                            { left: '\\(', right: '\\)', display: false },
                            { left: '\\[', right: '\\]', display: true }
                        ],
                        throwOnError: false
                    });
                }
            })
            .catch((error) => {
                console.error('Error loading markdown file:', error);
                renderError(outputEl, 'Error loading note content. Please ensure the Markdown file exists and you are running on a local server (or GitHub Pages).');
            });
    }

    window.renderMarkdownNote = renderMarkdownNote;

    const urlMarkdown = getMarkdownFromUrl();
    if (window.__NOTE_CONFIG__ || urlMarkdown) {
        renderMarkdownNote(window.__NOTE_CONFIG__);
    }
})();
