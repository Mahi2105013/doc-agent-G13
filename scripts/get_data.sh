#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

PDF_URL="https://archive.org/download/ar-raheequl-makhtoom-bangla/Ar_Raheequl_Makhtoom_Bangla.pdf"
ITEM="ar-raheequl-makhtoom-bangla"
FILENAME="Ar_Raheequl_Makhtoom_Bangla.pdf"

OUTDIR="data/raw/$ITEM"
OUTPATH="$OUTDIR/$FILENAME"
TXTPATH="${OUTPATH%.pdf}.txt"

TESS_LANG="ben+ara+eng"

# ============================================================
# 0. Install required Linux tools
# ============================================================

echo "======================================"
echo " 0. INSTALLING OCR/PDF TOOLS"
echo "======================================"

sudo apt update

sudo apt install -y \
    curl \
    ocrmypdf \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-ara

echo ""
echo "Installed tools:"
echo "  ocrmypdf:  $(command -v ocrmypdf)"
echo "  tesseract: $(command -v tesseract)"
echo "  pdftoppm:  $(command -v pdftoppm)"
echo "  pdftotext: $(command -v pdftotext)"
echo "  pdfinfo:   $(command -v pdfinfo)"

echo ""
echo "Tesseract languages:"
tesseract --list-langs

# ============================================================
# 1. Prepare directories
# ============================================================

mkdir -p "$OUTDIR"

# ============================================================
# 2. Download PDF
# ============================================================

echo ""
echo "======================================"
echo " 1. DOWNLOADING PDF"
echo "======================================"

if [ -s "$OUTPATH" ]; then
    echo "Already downloaded:"
    echo "  $OUTPATH"
else
    echo "Downloading $FILENAME..."

    curl \
        -L \
        -C - \
        --fail \
        --retry 3 \
        --retry-delay 5 \
        "$PDF_URL" \
        -o "$OUTPATH"

    echo "Download complete!"
fi

# ============================================================
# 3. OCR
# ============================================================

echo ""
echo "======================================"
echo " 2. OCR PIPELINE"
echo "======================================"

if [ -s "$TXTPATH" ]; then
    echo "Text already exists:"
    echo "  $TXTPATH"
else

    DONE=false

    # --------------------------------------------------------
    # Method A: OCRmyPDF
    # --------------------------------------------------------

    echo ""
    echo "Trying OCRmyPDF..."

    TMP_PDF="${OUTPATH%.pdf}.ocr.pdf"

    if ocrmypdf \
        --skip-text \
        -l "$TESS_LANG" \
        "$OUTPATH" \
        "$TMP_PDF"
    then

        echo "OCRmyPDF succeeded."

        if pdftotext "$TMP_PDF" "$TXTPATH"; then
            echo "Text extraction succeeded."
            rm -f "$TMP_PDF"
            DONE=true
        else
            echo "pdftotext failed."
            rm -f "$TMP_PDF"
        fi

    else
        echo "OCRmyPDF failed."
        rm -f "$TMP_PDF"
    fi

    # --------------------------------------------------------
    # Method B: pdftoppm + Tesseract fallback
    # --------------------------------------------------------

    if [ "$DONE" = false ]; then

        echo ""
        echo "Falling back to pdftoppm + Tesseract..."

        WORKDIR="${OUTPATH%.pdf}_pages"

        rm -rf "$WORKDIR"
        mkdir -p "$WORKDIR"

        echo "Converting PDF pages to PNG..."

        pdftoppm \
            -png \
            "$OUTPATH" \
            "$WORKDIR/page"

        : > "$TXTPATH"

        PAGE_COUNT=0
        FAILED_COUNT=0

        for img in "$WORKDIR"/page-*.png; do

            PAGE_COUNT=$((PAGE_COUNT + 1))

            base="${img%.png}"

            echo "OCR page $PAGE_COUNT: $(basename "$img")"

            if tesseract \
                "$img" \
                "$base" \
                -l "$TESS_LANG"
            then

                if [ -f "$base.txt" ]; then
                    cat "$base.txt" >> "$TXTPATH"
                    printf '\n\n' >> "$TXTPATH"
                fi

            else
                echo "WARNING: Tesseract failed on $(basename "$img")" >&2
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi

        done

        rm -rf "$WORKDIR"

        if [ "$FAILED_COUNT" -eq 0 ]; then
            DONE=true
            echo "Fallback OCR completed successfully."
        else
            echo "WARNING: $FAILED_COUNT pages failed OCR." >&2
        fi
    fi

    if [ "$DONE" = false ]; then
        echo ""
        echo "ERROR: OCR failed."
        exit 1
    fi
fi

# ============================================================
# 4. Summary
# ============================================================

echo ""
echo "======================================"
echo " 3. CORPUS SUMMARY"
echo "======================================"

ls -lh "$OUTDIR"

if command -v pdfinfo >/dev/null 2>&1; then
    PAGES=$(pdfinfo "$OUTPATH" | awk '/^Pages:/ {print $2}')
    echo ""
    echo "PDF pages: $PAGES"
fi

if [ -s "$TXTPATH" ]; then
    WORDS=$(wc -w < "$TXTPATH")
    SIZE=$(du -h "$TXTPATH" | cut -f1)

    echo "Text file: $TXTPATH"
    echo "Text size: $SIZE"
    echo "Extracted words: ~$WORDS"
else
    echo "WARNING: No text file produced."
fi

echo ""
echo "======================================"
echo " DONE"
echo "======================================"