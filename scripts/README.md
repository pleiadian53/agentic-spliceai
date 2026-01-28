# Scripts Directory

Utility scripts for the Agentic SpliceAI project.

## Available Scripts

### 📄 `md_to_pdf.py` - Markdown to PDF Converter

Convert Markdown documents to professional PDFs using Pandoc and XeLaTeX.

**Features**:
- ✅ Uses Pandoc (industry standard for document conversion)
- ✅ Generates publication-quality PDFs with XeLaTeX
- ✅ Automatic table of contents and section numbering
- ✅ Custom title and author metadata
- ✅ LaTeX-only mode for intermediate output
- ✅ Fallback to CLI if pypandoc not installed

**Requirements**:
- **Pandoc** (required): Document converter
  ```bash
  # macOS
  brew install pandoc
  
  # Linux
  sudo apt-get install pandoc
  ```

- **XeLaTeX** (required for PDF): LaTeX engine
  - See `docs/installation/LATEX_SETUP.md` for installation
  - Already installed at: `/usr/local/texlive/2025basic/bin/universal-darwin/xelatex`

- **pypandoc** (optional): Python wrapper for Pandoc
  ```bash
  pip install pypandoc
  ```

**Usage**:

```bash
# Basic conversion (generates PDF in same directory)
python scripts/md_to_pdf.py docs/API.md

# Specify output path
python scripts/md_to_pdf.py docs/API.md -o output/api_documentation.pdf

# With custom title and author
python scripts/md_to_pdf.py docs/BIOLOGY.md \
    --title "Splice Site Biology Primer" \
    --author "Agentic SpliceAI Team"

# Generate LaTeX only (for debugging or customization)
python scripts/md_to_pdf.py docs/TUTORIAL.md --latex-only
```

**Examples**:

```bash
# Convert README to PDF
python scripts/md_to_pdf.py README.md

# Convert research notes with metadata
python scripts/md_to_pdf.py dev/notes/research_summary.md \
    --title "Research Summary" \
    --author "Your Name" \
    -o output/research_summary.pdf

# Generate LaTeX for manual editing
python scripts/md_to_pdf.py docs/ARCHITECTURE.md --latex-only
# Creates: docs/ARCHITECTURE.tex
```

**Output Features**:
- 📑 Automatic table of contents
- 🔢 Numbered sections
- 🔗 Clickable hyperlinks (blue)
- 📏 1-inch margins
- 📄 Professional formatting

**Troubleshooting**:

| Issue | Solution |
|-------|----------|
| `pandoc: command not found` | Install Pandoc: `brew install pandoc` |
| `xelatex: command not found` | Add to PATH: See `docs/installation/LATEX_SETUP.md` |
| PDF generation fails | Try `--latex-only` first to debug LaTeX errors |
| pypandoc not found | Optional - script will use CLI fallback |

**Technical Details**:

The script uses Pandoc with these options:
- `--pdf-engine=xelatex` - Use XeLaTeX for PDF generation
- `--toc` - Generate table of contents
- `--number-sections` - Number all sections
- `-V geometry:margin=1in` - Set 1-inch margins
- `-V colorlinks=true` - Enable colored hyperlinks
- `-V linkcolor=blue` - Blue links for readability

## Data Scripts

### `data/` Directory

Contains data-related setup scripts (private, not in git).

See `tests/data/README.md` for data setup scripts.

## Adding New Scripts

When adding new scripts to this directory:

1. **Make executable**: `chmod +x scripts/your_script.sh`
2. **Add shebang**: `#!/usr/bin/env python3` or `#!/bin/bash`
3. **Document here**: Add usage section to this README
4. **Add help text**: Include `--help` option in your script
5. **Test thoroughly**: Verify on clean environment

## Script Conventions

- **Python scripts**: Use `#!/usr/bin/env python3`
- **Shell scripts**: Use `#!/bin/bash`
- **Naming**: Use `snake_case.py` or `kebab-case.sh`
- **Location**: Public scripts go in `scripts/`, private in `tests/`
- **Documentation**: Update this README with usage examples

## Related Documentation

- `docs/installation/LATEX_SETUP.md` - LaTeX installation guide
- `tests/data/README.md` - Data setup scripts (private)
- `CONTRIBUTING.md` - Directory conventions

---

**Need help?** Check the script's built-in help:
```bash
python scripts/md_to_pdf.py --help
```
