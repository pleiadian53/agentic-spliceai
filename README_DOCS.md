# Documentation Guide

This project uses MkDocs with Material theme to generate beautiful documentation with LaTeX math rendering support.

## 📚 Viewing the Documentation

**Live Site**: [https://pleiadian53.github.io/agentic-spliceai/](https://pleiadian53.github.io/agentic-spliceai/)

## 🔧 Local Development

### Prerequisites

The documentation dependencies are included in the `environment.yml` file. If you need to install them separately:

```bash
pip install -r requirements-docs.txt
```

### Serving Locally

```bash
# Activate the environment
mamba activate agentic-spliceai

# Start the development server
mkdocs serve

# View at: http://127.0.0.1:8000/agentic-spliceai/
```

### Building Documentation

```bash
# Build the static site
mkdocs build

# Build with strict mode (fails on warnings)
mkdocs build --strict
```

## 🚀 Deployment

### Automatic Deployment

Documentation is automatically deployed to GitHub Pages whenever you push to the `main` branch via GitHub Actions (`.github/workflows/docs.yml`).

**Workflow**: Push to main → GitHub Actions builds → Deploy to gh-pages → Site updates (2-3 minutes)

### Manual Deployment

```bash
# Deploy to GitHub Pages manually
mkdocs gh-deploy
```

## ✍️ Writing Documentation

### File Organization

- All documentation files go in the `docs/` folder
- Top-level files (README, QUICKSTART, SETUP, CONTRIBUTING) are copied to `docs/` for MkDocs
- Navigation is configured in `mkdocs.yml`

### Math Notation

#### Inline Math

```markdown
The equation $E = mc^2$ is famous.
```

#### Display Math

```markdown
$$
\mathcal{L}(X, Y) = \|R - XY^T\|_F^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)
$$
```

#### Multi-line Equations

```markdown
$$
\begin{align}
x_u &= (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u \\
y_i &= (X C_i X^T + \lambda I)^{-1} X C_i r_i
\end{align}
$$
```

### Code Blocks

````markdown
```python
def hello_world():
    print("Hello, world!")
```
````

### Admonitions (Callouts)

```markdown
!!! note "Note Title"
    This is a note with a custom title.

!!! warning
    This is a warning without a custom title.

!!! tip
    This is a helpful tip.
```

### Mermaid Diagrams

````markdown
```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```
````

## 🎨 Theme Features

- **Dark/Light Mode**: Toggle in the top navigation
- **Search**: Full-text search built-in
- **Code Copy**: Copy button on all code blocks
- **Navigation**: Tabs, sections, and expandable menus
- **Mobile Responsive**: Works great on all devices

## 📝 Configuration

### Main Config: `mkdocs.yml`

- **Site Information**: Name, URL, description
- **Theme Settings**: Colors, fonts, features
- **Navigation**: Page structure
- **Plugins**: Search, Jupyter support
- **Extensions**: Math, code highlighting, admonitions

### Custom Files

- `docs/javascripts/mathjax.js`: MathJax configuration for LaTeX rendering
- `docs/stylesheets/extra.css`: Custom CSS styles

## 🔍 Troubleshooting

### Math Not Rendering

1. Check browser console for errors (F12)
2. Verify MathJax is loaded: Type `MathJax` in console (should not be `undefined`)
3. Hard refresh: Ctrl+Shift+R (Cmd+Shift+R on Mac)

### Navigation Links 404

1. Ensure files are in `docs/` folder (not symlinks!)
2. Check paths in `mkdocs.yml` are relative to `docs/`
3. Rebuild: `mkdocs build`

### Site Not Updating After Push

1. Check GitHub Actions: [Actions Tab](https://github.com/pleiadian53/agentic-spliceai/actions)
2. Wait 2-3 minutes for deployment
3. Hard refresh browser: Ctrl+Shift+R
4. Check GitHub Pages is enabled: Settings → Pages

## 📦 Dependencies

Documentation dependencies are managed in two places:

1. **`requirements-docs.txt`**: For CI/CD (GitHub Actions)
2. **`environment.yml`**: For local development (included in pip section)

## 🔗 Useful Links

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MathJax Documentation](https://docs.mathjax.org/)
- [Markdown Guide](https://www.markdownguide.org/)
- [LaTeX Math Symbols](https://math.meta.stackexchange.com/questions/5020/)

## 📋 GitHub Pages Configuration

### First-Time Setup

1. Go to: [Repository Settings → Pages](https://github.com/pleiadian53/agentic-spliceai/settings/pages)
2. **Source**: Deploy from a branch
3. **Branch**: Select `gh-pages`
4. **Folder**: `/ (root)`
5. Click **Save**

The site should be live at: https://pleiadian53.github.io/agentic-spliceai/

---

For questions or issues, please refer to the [MkDocs documentation](https://www.mkdocs.org/) or open an issue.
