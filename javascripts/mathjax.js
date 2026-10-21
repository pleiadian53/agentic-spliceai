window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ["\\(", "\\)"]],
    displayMath: [['$$', '$$'], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process|arithmatex'
  },
  startup: {
    pageReady() {
      return MathJax.startup.defaultPageReady().then(() => {
        console.log('MathJax initial typesetting complete');
      });
    }
  }
};

// Re-render math when new content loads (for MkDocs Material theme)
document$.subscribe(() => { 
  MathJax.typesetPromise().catch((err) => console.log('MathJax typeset error:', err));
})
