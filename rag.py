"""Guideline retrieval (RAG): TF-IDF search over files in guidelines/. Runs on the Pi, no GPU needed."""
import glob
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cowtrol_common import path


def _read(f):
    if f.lower().endswith(".pdf"):
        try:
            import pypdf
        except ImportError:
            print(f"[warn] pypdf not installed; skipping {os.path.basename(f)}")
            return ""
        return "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(f).pages)
    with open(f, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _chunks(text, words=120):
    """Split on markdown headings / blank lines, then cap chunk length."""
    parts = [p.strip() for p in re.split(r"\n(?=#)|\n\s*\n", text) if p.strip()]
    out, cur = [], []
    for p in parts:
        w = p.split()
        if cur and len(cur) + len(w) > words:
            out.append(" ".join(cur))
            cur = []
        cur += w
        while len(cur) > words * 2:
            out.append(" ".join(cur[:words]))
            cur = cur[words:]
    if cur:
        out.append(" ".join(cur))
    return out


class Guidelines:
    def __init__(self, cfg):
        self.top_k = cfg["guidelines"]["top_k"]
        self.chunks = []  # (id, source, text)
        files = sorted(glob.glob(os.path.join(path(cfg["guidelines"]["dir"]), "*")))
        for f in files:
            if not f.lower().endswith((".md", ".txt", ".pdf")):
                continue
            for c in _chunks(_read(f)):
                self.chunks.append((f"G{len(self.chunks) + 1}", os.path.basename(f), c))
        self.vec = None
        if self.chunks:
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.mat = self.vec.fit_transform([c[2] for c in self.chunks])
        print(f"Guidelines: {len(self.chunks)} chunks from {len(files)} files")

    def search(self, query):
        if not self.vec:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        idx = sims.argsort()[::-1][: self.top_k]
        return [{"id": self.chunks[i][0], "source": self.chunks[i][1], "text": self.chunks[i][2]}
                for i in idx if sims[i] > 0]

    def ids(self):
        return {c[0] for c in self.chunks}
