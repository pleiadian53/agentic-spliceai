#!/bin/bash
# Complete renaming script: splice-agent → agentic-spliceai
# Run this from the splice-agent directory

set -e  # Exit on error

echo "=========================================="
echo "Renaming splice-agent → agentic-spliceai"
echo "=========================================="

# Step 1: Rename Python package directory
echo ""
echo "Step 1: Renaming splice_agent/ → agentic_spliceai/"
if [ -d "splice_agent" ]; then
    mv splice_agent agentic_spliceai
    echo "✓ Renamed directory"
else
    echo "⚠️  splice_agent/ directory not found"
fi

# Step 2: Rename egg-info directory if it exists
echo ""
echo "Step 2: Renaming egg-info directory"
if [ -d "splice_agent.egg-info" ]; then
    rm -rf splice_agent.egg-info
    echo "✓ Removed old egg-info (will be regenerated)"
else
    echo "✓ No egg-info to remove"
fi

# Step 3: Update pyproject.toml
echo ""
echo "Step 3: Updating pyproject.toml"
sed -i.bak 's/name = "splice-agent"/name = "agentic-spliceai"/g' pyproject.toml
sed -i.bak 's/splice-agent\[/agentic-spliceai[/g' pyproject.toml
sed -i.bak 's/splice-agent = "splice_agent/agentic-spliceai = "agentic_spliceai/g' pyproject.toml
sed -i.bak 's/splice-server = "splice_agent/agentic-spliceai-server = "agentic_spliceai/g' pyproject.toml
sed -i.bak 's/yourusername\/splice-agent/pleiadian53\/agentic-spliceai/g' pyproject.toml
sed -i.bak 's/"splice_agent/"agentic_spliceai/g' pyproject.toml
echo "✓ Updated pyproject.toml"

# Step 4: Update environment.yml
echo ""
echo "Step 4: Updating environment.yml"
sed -i.bak 's/name: splice-agent/name: agentic-spliceai/g' environment.yml
echo "✓ Updated environment.yml"

# Step 5: Update all Python imports
echo ""
echo "Step 5: Updating Python imports in all .py files"
find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.venv/*" ! -path "./env/*" -exec sed -i.bak 's/from splice_agent/from agentic_spliceai/g' {} \;
find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.venv/*" ! -path "./env/*" -exec sed -i.bak 's/import splice_agent/import agentic_spliceai/g' {} \;
find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.venv/*" ! -path "./env/*" -exec sed -i.bak 's/splice_agent\./agentic_spliceai./g' {} \;
echo "✓ Updated Python imports"

# Step 6: Update documentation
echo ""
echo "Step 6: Updating documentation files"
for file in README.md QUICKSTART.md MIGRATION.md STRUCTURE.md; do
    if [ -f "$file" ]; then
        sed -i.bak 's/splice-agent/agentic-spliceai/g' "$file"
        sed -i.bak 's/splice_agent/agentic_spliceai/g' "$file"
        echo "✓ Updated $file"
    fi
done

# Update docs directory
find docs -name "*.md" -type f -exec sed -i.bak 's/splice-agent/agentic-spliceai/g' {} \; 2>/dev/null || true
find docs -name "*.md" -type f -exec sed -i.bak 's/splice_agent/agentic_spliceai/g' {} \; 2>/dev/null || true
echo "✓ Updated docs/"

# Step 7: Clean up backup files
echo ""
echo "Step 7: Cleaning up backup files"
find . -name "*.bak" -type f -delete
echo "✓ Removed .bak files"

# Step 8: Summary
echo ""
echo "=========================================="
echo "✅ Renaming Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Test imports: python -c 'import agentic_spliceai'"
echo "3. Reinstall package: pip install -e ."
echo "4. Update conda env: mamba env update -f environment.yml"
echo "5. Test CLI: agentic-spliceai --help"
echo "6. Rename directory: cd .. && mv splice-agent agentic-spliceai"
echo "7. Create GitHub repo: github.com/pleiadian53/agentic-spliceai"
echo ""
