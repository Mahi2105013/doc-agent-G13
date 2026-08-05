#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# Downloads files for the Archive.org item `ar-raheequl-makhtoom-bangla`
set -euo pipefail

ITEM="ar-raheequl-makhtoom-bangla"
OUTDIR="data/raw/$ITEM"
mkdir -p "$OUTDIR"

echo "Fetching metadata for $ITEM..."
if ! command -v curl >/dev/null 2>&1; then
	echo "curl is required but not installed. Please install curl or download the item manually." >&2
	exit 1
fi

curl -s -L "https://archive.org/metadata/$ITEM" -o "$OUTDIR/metadata.json"

echo "Parsing metadata and downloading selected files into $OUTDIR"
python - <<'PY'
import json, os, subprocess, urllib.parse

item = "ar-raheequl-makhtoom-bangla"
outdir = os.path.join('data', 'raw', item)
meta_path = os.path.join(outdir, 'metadata.json')
with open(meta_path, 'r', encoding='utf-8') as f:
	meta = json.load(f)

files = meta.get('files', [])
wanted_exts = ('.pdf', '.djvu', '.txt', '.zip', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

for f in files:
	name = f.get('name')
	if not name:
		continue
	lname = name.lower()
	if lname.endswith(wanted_exts) or f.get('format') in ('PDF', 'DjVu', 'Text', 'TXT', 'JPEG', 'PNG', 'TIFF'):
		url = f"https://archive.org/download/{item}/{urllib.parse.quote(name)}"
		outpath = os.path.join(outdir, name)
		os.makedirs(os.path.dirname(outpath), exist_ok=True)
		if os.path.exists(outpath):
			print('Skipping existing', name)
			continue
		print('Downloading', name)
		subprocess.check_call(['curl', '-L', '-C', '-', '-o', outpath, url])

print('Download script finished.')
PY

echo "Files downloaded to $OUTDIR"

echo "Starting OCR pass for downloaded PDFs (if tools available)..."
python - <<'PY'
import os, subprocess, shutil, sys
from pathlib import Path

item = 'ar-raheequl-makhtoom-bangla'
outdir = Path('data') / 'raw' / item
pdfs = list(outdir.rglob('*.pdf'))
if not pdfs:
	print('No PDFs found for OCR.')
	sys.exit(0)

ocrmypdf = shutil.which('ocrmypdf')
pdftotext = shutil.which('pdftotext')
pdftoppm = shutil.which('pdftoppm')
tesseract = shutil.which('tesseract')

TESS_LANG = os.environ.get('TESSERACT_LANG', 'ben')

def run_ocrmypdf(pdf_path, txt_path):
	tmp_pdf = str(pdf_path.with_suffix('.ocr.pdf'))
	try:
		subprocess.check_call(['ocrmypdf', '--skip-text', str(pdf_path), tmp_pdf])
		if pdftotext:
			subprocess.check_call(['pdftotext', tmp_pdf, str(txt_path)])
			return True
	except Exception as e:
		print('ocrmypdf failed for', pdf_path, e)
	return False

def run_pdftoppm_tesseract(pdf_path, txt_path):
	# Convert pages to images and run tesseract on each
	try:
		workdir = pdf_path.with_suffix('')
		workdir = workdir.parent / (workdir.name + '_pages')
		workdir.mkdir(parents=True, exist_ok=True)
		prefix = str(workdir / 'page')
		subprocess.check_call(['pdftoppm', '-png', str(pdf_path), prefix])
		parts = sorted(workdir.glob('page-*.png')) + sorted(workdir.glob('page*.png'))
		texts = []
		for i, img in enumerate(parts):
			out_txt_p = workdir / f'page_{i}.txt'
			lang = TESS_LANG
			subprocess.check_call(['tesseract', str(img), str(out_txt_p.with_suffix('')), '-l', lang])
			with open(out_txt_p, 'r', encoding='utf-8', errors='ignore') as f:
				texts.append(f.read())
		with open(txt_path, 'w', encoding='utf-8') as f:
			f.write('\n\n'.join(texts))
		return True
	except Exception as e:
		print('pdftoppm+tesseract failed for', pdf_path, e)
		return False

for pdf in pdfs:
	txt_path = pdf.with_suffix('.txt')
	if txt_path.exists():
		print('Skipping OCR; text exists for', pdf.name)
		continue
	print('OCRing', pdf.name)
	done = False
	if ocrmypdf:
		done = run_ocrmypdf(pdf, txt_path)
	if not done and pdftoppm and tesseract:
		done = run_pdftoppm_tesseract(pdf, txt_path)
	if not done:
		print('Skipping', pdf.name, '- no OCR toolchain available or OCR failed.')

print('OCR pass complete.')
PY
