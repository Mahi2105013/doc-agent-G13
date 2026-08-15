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
DONE_MARKER="${TXTPATH}.complete"

# Baseline OCR languages.
TESS_LANG="ben+ara+eng"

# Rendering resolution.
DPI=300

# ============================================================
# 0. Install required tools
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
    tesseract-ocr-ara \
    tesseract-ocr-eng

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

# Verify required languages.
for lang in ben ara eng; do
    if ! tesseract --list-langs 2>/dev/null | grep -qx "$lang"; then
        echo "ERROR: Required Tesseract language '$lang' is missing." >&2
        exit 1
    fi
done

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

# If a complete/valid corpus already exists, skip everything.
if [ -s "$TXTPATH" ] && [ -f "$DONE_MARKER" ]; then
    echo "Complete OCR corpus already exists:"
    echo "  $TXTPATH"
else

    # Validate existing PDF before trusting it.
    if [ -s "$OUTPATH" ] && pdfinfo "$OUTPATH" >/dev/null 2>&1; then
        echo "Valid PDF already exists:"
        echo "  $OUTPATH"
    else

        if [ -f "$OUTPATH" ]; then
            echo "Existing PDF appears invalid/incomplete."
            echo "Removing it before re-downloading..."
            rm -f "$OUTPATH"
        fi

        echo "Downloading $FILENAME..."

        TMP_DOWNLOAD="${OUTPATH}.part"

        curl \
            -L \
            --fail \
            --retry 5 \
            --retry-delay 5 \
            --retry-all-errors \
            "$PDF_URL" \
            -o "$TMP_DOWNLOAD"

        # Validate downloaded PDF.
        if ! pdfinfo "$TMP_DOWNLOAD" >/dev/null 2>&1; then
            echo "ERROR: Downloaded file is not a valid PDF." >&2
            rm -f "$TMP_DOWNLOAD"
            exit 1
        fi

        mv "$TMP_DOWNLOAD" "$OUTPATH"

        echo "Download complete and PDF validated."
    fi
fi

# ============================================================
# 3. OCR
# ============================================================

echo ""
echo "======================================"
echo " 2. OCR PIPELINE"
echo "======================================"

if [ -s "$TXTPATH" ] && [ -f "$DONE_MARKER" ]; then

    echo "Complete text already exists:"
    echo "  $TXTPATH"

else

    rm -f "$TXTPATH" "$DONE_MARKER"

    DONE=false

    # --------------------------------------------------------
    # Method A: OCRmyPDF
    # --------------------------------------------------------

    echo ""
    echo "Trying OCRmyPDF..."

    TMP_PDF="${OUTPATH%.pdf}.ocr.pdf"

    rm -f "$TMP_PDF"

    if ocrmypdf \
        --force-ocr \
        -l "$TESS_LANG" \
        "$OUTPATH" \
        "$TMP_PDF"
    then

        echo "OCRmyPDF succeeded."

        if pdftotext "$TMP_PDF" "$TXTPATH" &&
           [ -s "$TXTPATH" ]; then

            WORDS=$(wc -w < "$TXTPATH")

            if [ "$WORDS" -gt 100 ]; then
                echo "Text extraction succeeded."
                echo "Extracted words: ~$WORDS"

                rm -f "$TMP_PDF"

                touch "$DONE_MARKER"
                DONE=true
            else
                echo "OCRmyPDF produced suspiciously little text."
                rm -f "$TXTPATH"
            fi

        else
            echo "pdftotext failed or produced no text."
            rm -f "$TXTPATH"
        fi

    else
        echo "OCRmyPDF failed."
    fi

    rm -f "$TMP_PDF"

    # --------------------------------------------------------
    # Method B: pdftoppm + Tesseract fallback
    # --------------------------------------------------------

    if [ "$DONE" = false ]; then

        echo ""
        echo "Falling back to pdftoppm + Tesseract..."

        WORKDIR="${OUTPATH%.pdf}_pages"

        rm -rf "$WORKDIR"
        mkdir -p "$WORKDIR"

        echo "Converting PDF pages to images at ${DPI} DPI..."

        pdftoppm \
            -png \
            -r "$DPI" \
            "$OUTPATH" \
            "$WORKDIR/page"

        # Safely collect generated pages.
        mapfile -t IMAGES < <(
            find "$WORKDIR" \
                -maxdepth 1 \
                -type f \
                -name 'page-*.png' \
                -print |
            sort -V
        )

        if [ "${#IMAGES[@]}" -eq 0 ]; then
            echo "ERROR: pdftoppm produced no page images." >&2
            rm -rf "$WORKDIR"
            exit 1
        fi

        echo "Generated ${#IMAGES[@]} page images."

        : > "$TXTPATH"

        PAGE_COUNT=0
        FAILED_COUNT=0

        for img in "${IMAGES[@]}"; do

            PAGE_COUNT=$((PAGE_COUNT + 1))

            base="${img%.png}"

            echo "OCR page $PAGE_COUNT/${#IMAGES[@]}: $(basename "$img")"

            if tesseract \
                "$img" \
                "$base" \
                -l "$TESS_LANG" \
                --psm 6
            then

                if [ -f "$base.txt" ]; then
                    cat "$base.txt" >> "$TXTPATH"
                    printf '\n\n' >> "$TXTPATH"
                else
                    echo "WARNING: No text output for $(basename "$img")" >&2
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi

            else
                echo "WARNING: Tesseract failed on $(basename "$img")" >&2
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi

        done

        WORDS=$(wc -w < "$TXTPATH")

        if [ "$FAILED_COUNT" -eq 0 ] && [ "$WORDS" -gt 100 ]; then

            DONE=true
            touch "$DONE_MARKER"

            echo ""
            echo "Fallback OCR completed successfully."
            echo "Pages processed: $PAGE_COUNT"
            echo "Extracted words: ~$WORDS"

        else

            echo ""
            echo "ERROR: Fallback OCR did not complete successfully." >&2
            echo "Pages processed: $PAGE_COUNT"
            echo "Failed pages: $FAILED_COUNT"
            echo "Extracted words: ~$WORDS" >&2

            rm -f "$TXTPATH" "$DONE_MARKER"

        fi

        rm -rf "$WORKDIR"
    fi

    if [ "$DONE" = false ]; then
        echo ""
        echo "ERROR: OCR failed." >&2
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