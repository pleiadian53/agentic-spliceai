#!/usr/bin/env python3
"""
Markdown to PDF Converter with Mermaid Support

Converts markdown files to PDF using Pandoc, with automatic preprocessing
of Mermaid diagrams to images.

Installation:
    pip install pypandoc  # Optional, for better error handling
    
    # Pandoc (required)
    brew install pandoc  # macOS
    sudo apt-get install pandoc  # Linux
    
    # Mermaid CLI (for diagram rendering)
    npm install -g @mermaid-js/mermaid-cli

Usage:
    python scripts/md_to_pdf.py docs/example.md
    python scripts/md_to_pdf.py docs/example.md --output output/example.pdf
    python scripts/md_to_pdf.py docs/example.md --no-mermaid  # Skip Mermaid preprocessing
"""

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False


# =============================================================================
# Mermaid Preprocessing
# =============================================================================

class MermaidConverter:
    """Convert Mermaid diagrams to images."""
    
    @staticmethod
    def check_mmdc() -> bool:
        """Check if mermaid-cli (mmdc) is available."""
        try:
            result = subprocess.run(['mmdc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def extract_mermaid_blocks(content: str) -> List[Tuple[int, int, str]]:
        """
        Extract Mermaid code blocks from markdown.
        
        Returns:
            List of (start_pos, end_pos, mermaid_code) tuples
        """
        pattern = r'```mermaid\n(.*?)```'
        matches = []
        for match in re.finditer(pattern, content, re.DOTALL):
            start = match.start()
            end = match.end()
            code = match.group(1).strip()
            matches.append((start, end, code))
        return matches
    
    @staticmethod
    def hash_mermaid(code: str) -> str:
        """Generate hash for Mermaid code (for caching)."""
        return hashlib.md5(code.encode()).hexdigest()[:8]
    
    @staticmethod
    def convert_diagram(mermaid_code: str, output_path: Path, 
                       debug: bool = False, use_png: bool = True) -> bool:
        """
        Convert Mermaid to image using mermaid-cli.
        
        Args:
            mermaid_code: Mermaid diagram code
            output_path: Path to save image (extension will be adjusted)
            debug: Print debug output on failure
            use_png: Use PNG format (recommended for PDF compatibility)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mermaid_code)
            temp_input = Path(f.name)
        
        # Force PNG extension for better PDF compatibility
        if use_png:
            output_path = output_path.with_suffix('.png')
        
        try:
            cmd = [
                'mmdc', 
                '-i', str(temp_input), 
                '-o', str(output_path),
                '-b', 'white',  # White background
                '-s', '1.5',  # Scale factor for quality (reduced from 2)
                '-w', '600',  # Max width in pixels
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if debug and result.returncode != 0:
                print(f"\n    mmdc stderr: {result.stderr}", file=sys.stderr)
            return result.returncode == 0 and output_path.exists()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if debug:
                print(f"\n    mmdc exception: {e}", file=sys.stderr)
            return False
        finally:
            temp_input.unlink(missing_ok=True)


def preprocess_mermaid(input_path: Path, output_path: Path,
                      image_dir: Optional[Path] = None,
                      debug: bool = False) -> Tuple[int, int]:
    """
    Preprocess markdown file to convert Mermaid diagrams to images.
    
    Args:
        input_path: Input markdown file
        output_path: Output markdown file with images
        image_dir: Directory to save images (default: output_dir/mermaid_images)
        debug: Print debug output on failure
        
    Returns:
        Tuple of (total_diagrams, successful_conversions)
    """
    content = input_path.read_text()
    mermaid_blocks = MermaidConverter.extract_mermaid_blocks(content)
    
    if not mermaid_blocks:
        # No Mermaid diagrams, just copy the file
        output_path.write_text(content)
        return 0, 0
    
    # Create image directory
    if image_dir is None:
        image_dir = output_path.parent / 'mermaid_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each Mermaid block
    replacements = []
    successful = 0
    
    for i, (start, end, mermaid_code) in enumerate(mermaid_blocks):
        # Generate image filename (use .png for PDF compatibility)
        diagram_hash = MermaidConverter.hash_mermaid(mermaid_code)
        image_filename = f"mermaid_{i+1}_{diagram_hash}.png"
        image_path = image_dir / image_filename
        
        # Convert Mermaid to image
        print(f"    Converting diagram {i+1}/{len(mermaid_blocks)}...", end=' ')
        if MermaidConverter.convert_diagram(mermaid_code, image_path, debug, use_png=True):
            print("✓")
            successful += 1
            # Use absolute path for Pandoc
            # The output path might have been adjusted to .png
            actual_image_path = image_path.with_suffix('.png')
            abs_image_path = actual_image_path.resolve()
            # Replace code block with image reference
            # Use Pandoc's width attribute to constrain image size
            image_markdown = f'![Mermaid diagram]({abs_image_path}){{ width=60% }}'
            replacements.append((start, end, image_markdown))
        else:
            print("✗")
            # Keep original code block if conversion fails
            replacements.append((start, end, content[start:end]))
    
    # Apply replacements in reverse order to preserve indices
    replacements.sort(key=lambda x: x[0], reverse=True)
    new_content = content
    for start, end, replacement in replacements:
        new_content = new_content[:start] + replacement + new_content[end:]
    
    output_path.write_text(new_content)
    return len(mermaid_blocks), successful


# =============================================================================
# Pandoc Conversion
# =============================================================================

class PandocConverter:
    """Markdown to PDF converter using Pandoc."""
    
    @staticmethod
    def check_pandoc() -> Tuple[bool, Optional[str]]:
        """Check if pandoc is installed."""
        try:
            result = subprocess.run(['pandoc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                return True, version
            return False, None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, None
    
    @staticmethod
    def check_xelatex() -> bool:
        """Check if XeLaTeX is installed."""
        try:
            result = subprocess.run(['xelatex', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def convert_to_latex(input_file: Path, output_file: Path, 
                        title: str = "", author: str = "") -> Tuple[bool, Optional[str]]:
        """Convert markdown to LaTeX using Pandoc."""
        if not PYPANDOC_AVAILABLE:
            return PandocConverter._convert_to_latex_cli(
                input_file, output_file, title, author
            )
        
        try:
            extra_args = [
                '--standalone',
                '--toc',
                '--number-sections',
                '-V', 'geometry:margin=1in',
                '-V', 'colorlinks=true',
                '-V', 'linkcolor=blue',
                '-V', 'urlcolor=blue',
            ]
            
            if title:
                extra_args.extend(['-V', f'title={title}'])
            if author:
                extra_args.extend(['-V', f'author={author}'])
            
            pypandoc.convert_file(
                str(input_file),
                'latex',
                outputfile=str(output_file),
                extra_args=extra_args
            )
            
            return True, None
            
        except Exception as e:
            return False, f"Pandoc conversion failed: {str(e)}"
    
    @staticmethod
    def _convert_to_latex_cli(input_file: Path, output_file: Path,
                              title: str = "", author: str = "") -> Tuple[bool, Optional[str]]:
        """Convert using pandoc CLI (fallback when pypandoc not available)."""
        cmd = [
            'pandoc',
            str(input_file),
            '-o', str(output_file),
            '--standalone',
            '--toc',
            '--number-sections',
            '-V', 'geometry:margin=1in',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
        ]
        
        if title:
            cmd.extend(['-V', f'title={title}'])
        if author:
            cmd.extend(['-V', f'author={author}'])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True, None
            return False, f"Pandoc error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Pandoc conversion timed out"
        except Exception as e:
            return False, f"Pandoc conversion failed: {str(e)}"
    
    @staticmethod
    def convert_to_pdf(input_file: Path, output_file: Path,
                      title: str = "", author: str = "") -> Tuple[bool, Optional[str]]:
        """Convert markdown directly to PDF using Pandoc."""
        if not PYPANDOC_AVAILABLE:
            return PandocConverter._convert_to_pdf_cli(
                input_file, output_file, title, author
            )
        
        try:
            extra_args = [
                '--pdf-engine=xelatex',
                '--toc',
                '--number-sections',
                '-V', 'geometry:margin=1in',
                '-V', 'colorlinks=true',
                '-V', 'linkcolor=blue',
                '-V', 'urlcolor=blue',
            ]
            
            if title:
                extra_args.extend(['-V', f'title={title}'])
            if author:
                extra_args.extend(['-V', f'author={author}'])
            
            pypandoc.convert_file(
                str(input_file),
                'pdf',
                outputfile=str(output_file),
                extra_args=extra_args
            )
            
            return True, None
            
        except Exception as e:
            return False, f"PDF generation failed: {str(e)}"
    
    @staticmethod
    def _convert_to_pdf_cli(input_file: Path, output_file: Path,
                           title: str = "", author: str = "") -> Tuple[bool, Optional[str]]:
        """Convert to PDF using pandoc CLI (fallback)."""
        cmd = [
            'pandoc',
            str(input_file),
            '-o', str(output_file),
            '--pdf-engine=xelatex',
            '--toc',
            '--number-sections',
            '-V', 'geometry:margin=1in',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
        ]
        
        if title:
            cmd.extend(['-V', f'title={title}'])
        if author:
            cmd.extend(['-V', f'author={author}'])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return True, None
            return False, f"PDF generation error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "PDF generation timed out"
        except Exception as e:
            return False, f"PDF generation failed: {str(e)}"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Markdown to PDF using Pandoc with Mermaid support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to PDF (auto-detects and converts Mermaid diagrams)
  python scripts/md_to_pdf.py docs/example.md
  
  # Specify output path
  python scripts/md_to_pdf.py docs/example.md -o output/example.pdf
  
  # Generate LaTeX only
  python scripts/md_to_pdf.py docs/example.md --latex-only
  
  # Skip Mermaid preprocessing
  python scripts/md_to_pdf.py docs/example.md --no-mermaid
  
  # Custom title and author
  python scripts/md_to_pdf.py docs/example.md --title "My Doc" --author "Name"

Requirements:
  - Pandoc: brew install pandoc (macOS) or sudo apt-get install pandoc (Linux)
  - XeLaTeX: For PDF generation
  - mermaid-cli: npm install -g @mermaid-js/mermaid-cli (for Mermaid diagrams)
  - pypandoc (optional): pip install pypandoc
        """
    )
    
    parser.add_argument('input', type=Path, help='Input markdown file')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output file path (default: same name with .pdf/.tex extension)')
    parser.add_argument('--latex-only', action='store_true',
                       help='Generate LaTeX file only, do not compile to PDF')
    parser.add_argument('--title', type=str, default="",
                       help='Document title')
    parser.add_argument('--author', type=str, default="",
                       help='Document author')
    parser.add_argument('--no-mermaid', action='store_true',
                       help='Skip Mermaid diagram preprocessing')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug output on failures')
    
    args = parser.parse_args()
    
    # Check dependencies
    pandoc_available, pandoc_version = PandocConverter.check_pandoc()
    if not pandoc_available:
        print("❌ Error: Pandoc is not installed", file=sys.stderr)
        print("\nInstallation instructions:", file=sys.stderr)
        print("  macOS:  brew install pandoc", file=sys.stderr)
        print("  Linux:  sudo apt-get install pandoc", file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ Found {pandoc_version}")
    
    if not args.latex_only:
        if not PandocConverter.check_xelatex():
            print("⚠️  Warning: XeLaTeX not found. PDF generation may fail.", file=sys.stderr)
    
    if not PYPANDOC_AVAILABLE:
        print("ℹ️  Note: pypandoc not installed, using pandoc CLI directly")
    
    # Check Mermaid CLI
    mermaid_available = MermaidConverter.check_mmdc()
    if mermaid_available:
        print("✓ Found mermaid-cli (mmdc)")
    else:
        print("ℹ️  mermaid-cli not found, Mermaid diagrams will not be rendered")
        print("   Install with: npm install -g @mermaid-js/mermaid-cli")
    
    # Validate input
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.latex_only:
            output_path = args.input.with_suffix('.tex')
        else:
            output_path = args.input.with_suffix('.pdf')
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Preprocess Mermaid diagrams if needed
    input_file = args.input
    temp_file = None
    
    if not args.no_mermaid and mermaid_available:
        # Check if file has Mermaid diagrams
        content = args.input.read_text()
        if '```mermaid' in content:
            print("🔄 Preprocessing Mermaid diagrams...")
            # Create temporary preprocessed file
            temp_file = args.input.parent / f".{args.input.stem}_preprocessed.md"
            image_dir = output_path.parent / 'mermaid_images'
            
            try:
                total, successful = preprocess_mermaid(
                    args.input, temp_file, image_dir, args.debug
                )
                if successful > 0:
                    print(f"✓ Converted {successful}/{total} Mermaid diagrams")
                    input_file = temp_file
                elif total > 0:
                    print(f"⚠️  All {total} Mermaid diagrams failed to convert")
                    print("   Using original file (diagrams will appear as code)")
                    temp_file = None
            except Exception as e:
                print(f"⚠️  Mermaid preprocessing failed: {e}")
                print("   Using original file...")
                temp_file = None
    
    # Convert
    print(f"📖 Reading: {input_file}")
    
    if args.latex_only:
        print("🔄 Converting to LaTeX...")
        success, error = PandocConverter.convert_to_latex(
            input_file, output_path, args.title, args.author
        )
        
        if success:
            print(f"✅ LaTeX generated: {output_path}")
            print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"❌ LaTeX generation failed:", file=sys.stderr)
            print(f"   {error}", file=sys.stderr)
            sys.exit(1)
    else:
        print("📄 Converting to PDF...")
        success, error = PandocConverter.convert_to_pdf(
            input_file, output_path, args.title, args.author
        )
        
        if success:
            print(f"✅ PDF generated: {output_path}")
            print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
            print(f"\n💡 Tip: Generate LaTeX with --latex-only to see intermediate output")
        else:
            print(f"❌ PDF generation failed:", file=sys.stderr)
            print(f"   {error}", file=sys.stderr)
            print(f"\n💡 Try generating LaTeX first: --latex-only", file=sys.stderr)
            sys.exit(1)
    
    # Clean up temporary file
    if temp_file and temp_file.exists():
        temp_file.unlink()


if __name__ == '__main__':
    main()
