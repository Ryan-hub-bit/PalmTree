#!/bin/bash
#
# Quick script to check if your raw corpus contains daddr(...) annotations
#

echo "================================================================================"
echo "Checking Raw Corpus for daddr(...) Annotations"
echo "================================================================================"
echo ""

# List of potential corpus paths to check
CORPUS_PATHS=(
    "/data/kun/jtrans/addressaware/corpus.txt"
    "/data/kun/jtrans/addressaware/train/corpus.txt"
    "/data/kun/jtrans/pretrain/corpus.txt"
    "/data/kun/corpus/addressaware.txt"
)

found_corpus=0

for corpus_path in "${CORPUS_PATHS[@]}"; do
    if [ -f "$corpus_path" ]; then
        echo "✅ Found corpus: $corpus_path"
        found_corpus=1
        
        # Check file size
        size=$(du -h "$corpus_path" | cut -f1)
        lines=$(wc -l < "$corpus_path")
        echo "   Size: $size, Lines: $lines"
        echo ""
        
        # Sample first 1000 lines
        echo "Checking first 1000 lines for address annotations..."
        address_count=$(head -1000 "$corpus_path" | grep -o "address(" | wc -l)
        daddr_count=$(head -1000 "$corpus_path" | grep -o "daddr(" | wc -l)
        var_count=$(head -1000 "$corpus_path" | grep -o "var(" | wc -l)
        
        echo "   'address(...)' occurrences: $address_count"
        echo "   'daddr(...)' occurrences:   $daddr_count"
        echo "   'var(...)' occurrences:     $var_count"
        echo ""
        
        if [ "$daddr_count" -gt 0 ]; then
            echo "✅ GOOD: Found daddr annotations in raw corpus!"
            echo "   → You can directly regenerate data with the fixed dataloader"
            echo ""
            echo "Sample daddr usage:"
            head -1000 "$corpus_path" | grep "daddr(" | head -3
        else
            echo "⚠️  WARNING: No daddr annotations found in raw corpus!"
            echo "   → Your data generator doesn't distinguish code vs data addresses"
            echo "   → Need to fix data generation pipeline before retraining"
            echo ""
            echo "Sample address usage:"
            head -1000 "$corpus_path" | grep "address(" | head -3
        fi
        echo ""
        echo "================================================================================"
        break
    fi
done

if [ $found_corpus -eq 0 ]; then
    echo "❌ No corpus file found in default locations."
    echo ""
    echo "Please specify your corpus path and run:"
    echo ""
    echo '  corpus_path="/path/to/your/corpus.txt"'
    echo '  echo "address count: $(grep -o \"address(\" \"$corpus_path\" | wc -l)"'
    echo '  echo "daddr count:   $(grep -o \"daddr(\" \"$corpus_path\" | wc -l)"'
    echo '  echo "var count:     $(grep -o \"var(\" \"$corpus_path\" | wc -l)"'
    echo ""
    echo "Sample paths to check:"
    for path in "${CORPUS_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo "================================================================================"
fi
