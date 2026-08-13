Document Processing Pipeline
----------------------------
[Loader] --> [Preprocessor] --> [Enhancer] --> [Layout Detector] --> [OCR] --> [Chunker] --> [Embedder] --> [Vector Store]

Detailed Flow:
1. loader.load_pages
   Reads raw documents and outputs Page objects.
   |
   v
2. preprocess.run
   Applies image preprocessing (e.g. binarization).
   |
   v
3. enhance.run (Stage 1)
   We did not attempt any enhancements. But the pipeline requires running it.
   |
   v
4. layout.detect (Stage 2)
   Uses YOLOv8 Nano (YOLOv8n) model that has been fine-tuned on the DocLayNet dataset. It is specialized for Document Layout Analysis to detect document regions of headers and footers for our use case.
   |
   v
5. ocr.transcribe (Stage 3)
   Extracts text computationally from the isolated layout regions.
   |
   v
6. chunk.split (Stage 4)
   - Takes extracted text and splits it into manageable pieces.
   - Uses recursive character boundary matching ('\n\n' -> '\n' -> ' ' -> '').
   - Basic defaults: chunk_size=500, chunk_overlap=50.
   - Outputs a list of Chunk objects enriched with chunk_index and origin metadata.
   |
   v
7. embed.encode (Stage 4)
   - Extracts the textual content from chunks.
   - Embeds vectors into high-dimensional space (default: all-MiniLM-L6-v2 via sentence-transformers).
   - Graceful fallback: TF-IDF vectorizer zero-padded to 384-dimensions if heavy neural frameworks are absent.
   - Outputs a normalized float32 numpy array.
   |
   v
8. store.build (Stage 4)
   - Initializes the local index directory.
   - Initializes vector store index (Faiss) and adds the embedding matrix.
   - Smart fallback: supports flat IP index natively, IVF (for larger document sets), and HNSW.
   - If Faiss is unavailable, reverts to a pure numpy `.npy` backup structure.
   - Persists the Chunk metadata via python pickle (`chunks.pkl`) so they correspond directly to index rows.
