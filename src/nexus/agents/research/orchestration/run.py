import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

from nexus.agents.research.orchestration import pipeline
from nexus.agents.research.provenance import manifest
from nexus.agents.research.utils import slug_utils
from nexus.agents.research.formatters import pdf_utils
from nexus.core.config import NexusConfig

def main():
    parser = argparse.ArgumentParser(description="AI Research Agent CLI")
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("--model", default="openai:gpt-4o", help="Model to use (e.g., openai:gpt-4o, openai:gpt-5.1-codex-mini)")
    parser.add_argument("--length", default="standard", 
                       choices=["brief", "standard", "comprehensive", "technical-paper"],
                       help="Target report length: brief (2-3 pages), standard (5-10 pages), comprehensive (15-25 pages), technical-paper (25-40 pages)")
    parser.add_argument("--context", help="Additional context or style template (e.g., 'Follow Nature Methods style')", default=None)
    parser.add_argument("--output", help="Path to save the final report", default=None)
    parser.add_argument("--pdf", action="store_true", help="Also generate PDF version of the report")
    parser.add_argument("--format", choices=["latex", "markdown", "pdf_direct"], 
                       help="Force specific output format (overrides automatic detection)", default=None)
    parser.add_argument("--verbose", action="store_true", default=True, 
                       help="Show detailed progress (default: True)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Minimal output (overrides --verbose)")
    
    args = parser.parse_args()
    
    # Determine verbose mode
    verbose = args.verbose and not args.quiet
    
    try:
        # Track generation time
        start_time = time.time()
        
        results = pipeline.generate_research_report(
            args.topic, 
            model=args.model, 
            report_length=args.length,
            context=args.context,
            user_format=args.format,
            verbose=verbose
        )
        
        generation_time = time.time() - start_time
        
        report = results.get("final_report", "")
        plan_steps = len(results.get("plan", []))
        
        if report:
            if verbose:
                print(f"\n{'='*70}")
                print(f"✅ REPORT GENERATION COMPLETE")
                print(f"{'='*70}\n")
            elif not args.quiet:
                print("\n✅ Report generated successfully")
            
            # Determine output location
            if args.output:
                output_path = Path(args.output)
            else:
                # Use unified output structure: output/research_reports/<topic_slug>/
                # Generate smart slug using LLM
                print("🏷️  Generating topic slug...")
                topic_slug = slug_utils.generate_topic_slug(args.topic, max_length=50)
                print(f"   → {topic_slug}")
                
                # Use NexusConfig for standardized path management
                topic_dir = NexusConfig.RESEARCH_REPORTS_DIR / topic_slug
                topic_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
                
                # Use appropriate file extension based on format
                format_info = results.get("format_decision", {})
                output_format = format_info.get("format", "markdown")
                file_extension = ".tex" if output_format == "latex" else ".md"
                filename = f"report_{timestamp}{file_extension}"
                output_path = topic_dir / filename
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            
            # Generate PDF if requested
            pdf_filename = None
            markdown_preview_filename = None
            format_info = results.get("format_decision", {})
            
            if args.pdf:
                print("\n📄 Generating PDF...")
                pdf_path = output_path.with_suffix('.pdf')
                
                # Use appropriate PDF generation method based on format
                output_format = format_info.get("format", "markdown")
                
                if output_format == "latex":
                    print(f"   Using LaTeX compilation (equations detected)")
                    success, error = pdf_utils.latex_to_pdf(
                        report, 
                        pdf_path,
                        title=args.topic
                    )
                    
                    # Also generate Markdown preview for GitHub
                    if success:
                        print(f"   Generating Markdown preview for GitHub...")
                        from ..formatters import pandoc_converter
                        md_preview_path = output_path.with_suffix('.md')
                        
                        # Try Pandoc first, fallback to custom converter
                        success_md, method = pandoc_converter.generate_markdown_preview(
                            report, md_preview_path, prefer_pandoc=True
                        )
                        
                        if success_md:
                            markdown_preview_filename = md_preview_path.name
                            converter_name = "Pandoc" if method == "pandoc" else "custom converter"
                            print(f"   ✓ Markdown preview saved: {md_preview_path} (using {converter_name})")
                        else:
                            print(f"   ⚠️  Markdown preview generation failed")
                else:
                    print(f"   Using Markdown→PDF conversion")
                    success, error = pdf_utils.markdown_to_pdf(
                        report, 
                        pdf_path,
                        title=args.topic
                    )
                
                if success:
                    pdf_filename = pdf_path.name
                    print(f"   ✓ PDF saved: {pdf_path}")
                else:
                    print(f"   ⚠️  PDF generation failed: {error}")
                    print(f"   ℹ️  Report still available at: {output_path}")
            
            # Create manifest entry
            topic_dir = output_path.parent
            manifest.create_manifest_entry(
                topic_dir=topic_dir,
                filename=output_path.name,
                topic=args.topic,
                model=args.model,
                report_length=args.length,
                context=args.context,
                source="cli",
                report_content=report,
                generation_time_seconds=round(generation_time, 2),
                plan_steps=plan_steps,
                pdf_filename=pdf_filename,
                format_decision=format_info  # Track format decision
            )
            
            print(f"\n📁 Saved to: {output_path}")
            if pdf_filename:
                print(f"📄 PDF: {output_path.parent / pdf_filename}")
            print(f"⏱️  Generation time: {generation_time:.1f}s")
            print(f"📊 Word count: ~{len(report.split())} words")
            print(f"📋 Plan steps: {plan_steps}")
            print(f"📝 Manifest updated: {topic_dir / 'manifest.json'}")
        else:
            print("❌ No report generated.")
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
