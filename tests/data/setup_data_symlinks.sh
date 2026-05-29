#!/bin/bash
# Setup data symlinks for agentic-spliceai
# Links large genomic datasets from meta-spliceai without copying

set -e  # Exit on error

echo "=========================================="
echo "Setting up data symlinks"
echo "=========================================="

# Source and target directories
SOURCE_DATA="/Users/pleiadian53/work/meta-spliceai/data"
TARGET_DATA="/Users/pleiadian53/work/agentic-spliceai/data"

# Ensure target data directory exists
mkdir -p "$TARGET_DATA"

echo ""
echo "Current agentic-spliceai data directory:"
ls -la "$TARGET_DATA"

echo ""
echo "=========================================="
echo "Creating symlinks to genomic datasets"
echo "=========================================="

# Key genomic datasets to symlink
GENOMIC_DIRS=(
    "ensembl"           # Ensembl annotations (GRCh37, GRCh38)
    "mane"              # MANE transcripts
    "GRCh38_MANE"       # GRCh38 MANE-specific
    "models"            # Pre-trained SpliceAI models
    "spliceai_analysis" # SpliceAI analysis results
    "spliceai_eval"     # SpliceAI evaluation data
)

# Create symlinks for genomic directories
for dir in "${GENOMIC_DIRS[@]}"; do
    SOURCE_PATH="$SOURCE_DATA/$dir"
    TARGET_PATH="$TARGET_DATA/$dir"
    
    if [ -d "$SOURCE_PATH" ]; then
        if [ -L "$TARGET_PATH" ]; then
            echo "⚠️  Symlink already exists: $dir"
        elif [ -d "$TARGET_PATH" ]; then
            echo "⚠️  Directory already exists (not a symlink): $dir"
            echo "    Skipping to avoid overwriting"
        else
            ln -s "$SOURCE_PATH" "$TARGET_PATH"
            echo "✓ Created symlink: $dir"
        fi
    else
        echo "⚠️  Source directory not found: $dir"
    fi
done

echo ""
echo "=========================================="
echo "Creating symlinks to training datasets"
echo "=========================================="

# Training/test datasets (potentially large)
TRAINING_DIRS=(
    "train_pc_1000_3mers"
    "train_pc_100_3mers_diverse"
    "test_pc_1000_3mers"
    "test_quick"
)

for dir in "${TRAINING_DIRS[@]}"; do
    SOURCE_PATH="$SOURCE_DATA/$dir"
    TARGET_PATH="$TARGET_DATA/$dir"
    
    if [ -d "$SOURCE_PATH" ]; then
        if [ -L "$TARGET_PATH" ]; then
            echo "⚠️  Symlink already exists: $dir"
        elif [ -d "$TARGET_PATH" ]; then
            echo "⚠️  Directory already exists (not a symlink): $dir"
        else
            ln -s "$SOURCE_PATH" "$TARGET_PATH"
            echo "✓ Created symlink: $dir"
        fi
    else
        echo "⚠️  Source directory not found: $dir"
    fi
done

echo ""
echo "=========================================="
echo "Creating symlinks to reference files"
echo "=========================================="

# Individual reference files to symlink
REFERENCE_FILES=(
    "query_uorfs.fa"
    "query_uorfs.gtf"
    "gtex_uORFconnected_txs.csv"
    "gtex_uORFconnected_txs.w_utrs.csv"
    "supplementary-tables.xlsx"
)

for file in "${REFERENCE_FILES[@]}"; do
    SOURCE_PATH="$SOURCE_DATA/$file"
    TARGET_PATH="$TARGET_DATA/$file"
    
    if [ -f "$SOURCE_PATH" ]; then
        if [ -L "$TARGET_PATH" ]; then
            echo "⚠️  Symlink already exists: $file"
        elif [ -f "$TARGET_PATH" ]; then
            echo "⚠️  File already exists (not a symlink): $file"
        else
            ln -s "$SOURCE_PATH" "$TARGET_PATH"
            echo "✓ Created symlink: $file"
        fi
    else
        echo "⚠️  Source file not found: $file"
    fi
done

echo ""
echo "=========================================="
echo "Optional: ORF and Han1 datasets"
echo "=========================================="

# Optional datasets (uncomment if needed)
OPTIONAL_DIRS=(
    "ORF"
    "Han1"
)

for dir in "${OPTIONAL_DIRS[@]}"; do
    SOURCE_PATH="$SOURCE_DATA/$dir"
    TARGET_PATH="$TARGET_DATA/$dir"
    
    if [ -d "$SOURCE_PATH" ]; then
        if [ -L "$TARGET_PATH" ]; then
            echo "⚠️  Symlink already exists: $dir"
        elif [ -d "$TARGET_PATH" ]; then
            echo "⚠️  Directory already exists: $dir"
        else
            ln -s "$SOURCE_PATH" "$TARGET_PATH"
            echo "✓ Created symlink: $dir"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Final data directory structure"
echo "=========================================="
echo ""
echo "Existing files (agentic-spliceai):"
ls -lh "$TARGET_DATA" | grep -v "^l" | grep -v "^d" || echo "  (none)"

echo ""
echo "Symlinked directories:"
ls -lh "$TARGET_DATA" | grep "^l" || echo "  (none)"

echo ""
echo "=========================================="
echo "✓ Data symlinks setup complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Genomic datasets: linked from meta-spliceai"
echo "  - Existing data: preserved in agentic-spliceai"
echo "  - No data duplication"
echo ""
echo "To verify:"
echo "  ls -lh $TARGET_DATA"
