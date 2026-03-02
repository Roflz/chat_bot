import json
import os
import re
import http.client
import fitz
import time

from contextlib import contextmanager
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
MODEL = "qwen2.5-coder:3b"
# Safety: keep file context bounded so you don't accidentally dump huge docs
MAX_FILE_CHARS = 120_000  # adjust if needed
SYSTEM_RULES = """You are a local assistant running in an offline, secure environment.
Rules:
- Never suggest uploading code/documents to external services.
- Assume you have no internet access.
- Only use file content that the user explicitly provided via /file.
- If file content is incomplete, ask what section to focus on.
"""

def ollama_generate(prompt: str) -> str:
   conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=900)
   payload = {
       "model": MODEL,
       "prompt": prompt,
       "stream": True,             # streaming ON
       "keep_alive": "10m",        # keep model warm
       "options": {
           "temperature": 0.2,
           "num_ctx": 2048,        # speed lever
           "num_predict": 512      # cap output
       }
   }
   body = json.dumps(payload).encode("utf-8")
   headers = {"Content-Type": "application/json"}
   conn.request("POST", "/api/generate", body=body, headers=headers)
   resp = conn.getresponse()
   if resp.status != 200:
       data = resp.read()
       conn.close()
       return f"[ERROR] Ollama returned HTTP {resp.status}: {data[:500]!r}"
   full = []
   # Ollama streams NDJSON: one JSON object per line
   while True:
       line = resp.readline()
       if not line:
           break
       line = line.decode("utf-8", errors="replace").strip()
       if not line:
           continue
       obj = json.loads(line)
       chunk = obj.get("response", "")
       if chunk:
           print(chunk, end="", flush=True)
           full.append(chunk)
       if obj.get("done", False):
           break
   print("")  # newline after streaming output
   conn.close()
   return "".join(full)

def read_file(path: str) -> str:
   # Expand ~ and environment vars, normalize
   path = os.path.expandvars(os.path.expanduser(path))
   with open(path, "r", encoding="utf-8", errors="replace") as f:
       text = f.read()
   if len(text) > MAX_FILE_CHARS:
       text = text[:MAX_FILE_CHARS] + "\n\n[TRUNCATED]\n"
   return text

# ----------------------------
# Common helpers
# ----------------------------
def norm_text(s: str) -> str:
   """Normalize whitespace for LLM-friendly text."""
   return re.sub(r"\s+", " ", s).strip()

# ----------------------------
# PDF translation (PyMuPDF)
# ----------------------------
def translate_pdf_to_text(path: str) -> str:
   doc = fitz.open(path)
   out_lines = []
   out_lines.append(f"===== PDF: {path} =====\n")
   for page_num, page in enumerate(doc, start=1):
       out_lines.append(f"===== PAGE {page_num} =====\n")
       # Use block-based extraction for better reading order
       blocks = page.get_text("blocks")
       blocks = [b for b in blocks if len(b) >= 5 and b[4].strip()]
       blocks.sort(key=lambda b: (b[1], b[0]))  # sort by y, then x
       page_lines = []
       for b in blocks:
           text = b[4]
           for raw_line in text.splitlines():
               line = norm_text(raw_line)
               if line:
                   page_lines.append(line)
       if page_lines:
           out_lines.extend(page_lines)
       else:
           out_lines.append("[NO EXTRACTED TEXT]")
       out_lines.append("")  # spacer between pages
   doc.close()
   return "\n".join(out_lines)

# ----------------------------
# DOCX translation (python-docx)
# ----------------------------
def iter_docx_blocks(doc: Document):
   """Yield Paragraph and Table objects in true document order."""
   body = doc.element.body
   for child in body.iterchildren():
       if isinstance(child, CT_P):
           yield Paragraph(child, doc)
       elif isinstance(child, CT_Tbl):
           yield Table(child, doc)

def paragraph_tag(p: Paragraph) -> str:
   """Return a readable tag for paragraph type."""
   try:
       style_name = p.style.name if p.style else ""
   except Exception:
       style_name = ""
   if style_name.lower().startswith("heading"):
       m = re.search(r"(\d+)", style_name)
       level = m.group(1) if m else ""
       return f"HEADING {level}".strip()
   return "P"

def translate_docx_to_text(path: str) -> str:
   doc = Document(path)
   out_lines = []
   out_lines.append(f"===== DOCX: {path} =====\n")
   p_count = 0
   t_count = 0
   for block in iter_docx_blocks(doc):
       if isinstance(block, Paragraph):
           text = norm_text(block.text)
           if not text:
               continue
           p_count += 1
           tag = paragraph_tag(block)
           out_lines.append(f"[{tag} #{p_count}] {text}")
       elif isinstance(block, Table):
           t_count += 1
           out_lines.append(f"\n--- TABLE {t_count} ---")
           for r_i, row in enumerate(block.rows, start=1):
               cells = [norm_text(cell.text) for cell in row.cells]
               out_lines.append(f"[R{r_i}] " + " | ".join(cells))
           out_lines.append("")  # spacer after table
   out_lines.append("===== END DOCX =====")
   return "\n".join(out_lines)

# ----------------------------
# Unified entry point
# ----------------------------
def read_any_file_as_text(path: str) -> str:
   """
   Translate supported files into clean, LLM-friendly text.
   Supported:
   - .pdf   -> PyMuPDF
   - .docx  -> python-docx (paragraphs + tables, in order)
   - other  -> plain text read
   """
   path = os.path.expandvars(os.path.expanduser(path))
   lower = path.lower()
   if lower.endswith(".pdf"):
       return translate_pdf_to_text(path)
   if lower.endswith(".docx"):
       return translate_docx_to_text(path)
   # Fallback: treat as plain text
   with open(path, "r", encoding="utf-8", errors="replace") as f:
       return f.read()

DEBUG = True  # set False to silence debug output
@contextmanager
def timed(label: str):
   start = time.perf_counter()
   try:
       yield
   finally:
       end = time.perf_counter()
       if DEBUG:
           ms = (end - start) * 1000.0
           print(f"[t] {label}: {ms:,.1f} ms")

def dbg(msg: str):
   if DEBUG:
       print(f"[dbg] {msg}")

def tokenize_for_search(s: str) -> list[str]:
   # simple word tokenizer for keyword scoring
   s = s.lower()
   words = re.findall(r"[a-z0-9_]+", s)
   # drop tiny tokens that add noise
   return [w for w in words if len(w) >= 3]

def chunk_text(text: str, chunk_chars: int = 3500, overlap_chars: int = 400) -> list[str]:
   """
   Simple, stable chunker:
   - chunks by line boundaries when possible
   - maintains a small overlap
   """
   lines = text.splitlines()
   chunks = []
   cur = []
   cur_len = 0
   for ln in lines:
       # +1 for newline
       add_len = len(ln) + 1
       if cur_len + add_len > chunk_chars and cur:
           chunk = "\n".join(cur).strip()
           if chunk:
               chunks.append(chunk)
           # overlap: keep last overlap_chars from this chunk
           if overlap_chars > 0:
               tail = chunk[-overlap_chars:]
               cur = [tail]
               cur_len = len(tail)
           else:
               cur = []
               cur_len = 0
       cur.append(ln)
       cur_len += add_len
   final = "\n".join(cur).strip()
   if final:
       chunks.append(final)
   return chunks

def rank_chunks(question: str, chunks: list[str], top_k: int = 4) -> list[tuple[int, int]]:
   """
   Return list of (score, index) for top chunks using simple keyword overlap.
   """
   q_words = set(tokenize_for_search(question))
   scored = []
   for i, ch in enumerate(chunks):
       # cheap scoring: count distinct query terms present in chunk
       ch_low = ch.lower()
       score = sum(1 for w in q_words if w in ch_low)
       scored.append((score, i))
   scored.sort(reverse=True)
   # keep only chunks with some match; if none match, fall back to first chunk(s)
   best = [(s, i) for (s, i) in scored if s > 0][:top_k]
   if not best:
       best = [(0, i) for i in range(min(top_k, len(chunks)))]
   return best

def main():
   global MODEL
   print("Local Secure Chat (Ollama)")
   print(f"- Model: {MODEL}")
   print("- Commands:")
   print("   /file <path>   (attach a file's contents to context)")
   print("   /clear         (clear attached file context)")
   print("   /model <name>  (change model tag)")
   print("   /show          (show currently attached files)")
   print("   /quit")
   print("")
   attachments = []
   while True:
       try:
           user = input("you> ").strip()
       except (EOFError, KeyboardInterrupt):
           print("\nbye.")
           return
       if not user:
           continue
       if user.lower() in ("/quit", "/exit"):
           print("bye.")
           return
       if user.startswith("/model "):
           MODEL = user.split(" ", 1)[1].strip()
           print(f"[ok] model set to: {MODEL}")
           continue
       if user == "/show":
           if not attachments:
               print("[info] no files attached")
           else:
               print("[info] attached files:")
               for i, a in enumerate(attachments, start=1):
                   print(f"  {i}. {a['label']} ({len(a['text'])} chars)")
           continue
       if user == "/clear":
           attachments = []
           print("[ok] cleared all attachments")
           continue
       if user.startswith("/file "):
           path = user.split(" ", 1)[1].strip().strip('"')
           try:
               print(path)
               with timed(f"translate file {path}"):
                   text = read_any_file_as_text(path)
               dbg(f"translated chars: {len(text):,}")

               if len(text) > MAX_FILE_CHARS:
                   text = text[:MAX_FILE_CHARS] + "\n\n[TRUNCATED]\n"
                   dbg(f"truncated to MAX_FILE_CHARS={MAX_FILE_CHARS:,}")

               chunks = chunk_text(text, chunk_chars=3500, overlap_chars=400)
               attachments.append({
                   "label": path,
                   "text": text,
                   "chunks": chunks
               })
               print(
                   f"[ok] attached file: {path} ({len(text)} chars) | chunks: {len(chunks)} | total attached: {len(attachments)}")
           except Exception as e:
               print(f"[error] couldn't read file: {e}")
           continue
       with timed("build prompt"):
           prompt_parts = [SYSTEM_RULES]
           if attachments:
               prompt_parts.append("Attached files (selected excerpts):\n")
               for f_idx, a in enumerate(attachments, start=1):
                   ranked = rank_chunks(user, a["chunks"], top_k=4)
                   prompt_parts.append(f"[FILE {f_idx}] {a['label']}\n")
                   for score, ch_i in ranked:
                       prompt_parts.append(f"--- CHUNK {ch_i + 1} (score={score}) ---\n{a['chunks'][ch_i]}\n")
           prompt_parts.append(f"User question:\n{user}\n\nAnswer:")
           prompt = "\n".join(prompt_parts)
       dbg(f"prompt chars: {len(prompt):,}")
       with timed("ollama_generate total"):
           answer = ollama_generate(prompt)
       print(f"assistant> {answer}\n")
if __name__ == "__main__":
   main()