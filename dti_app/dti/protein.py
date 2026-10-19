"""Protein sequence -> 320-d ESM-2 (8M) embedding.  embed logic mirrors training exactly:
windows of 900 residues, stride 800, mean over real residues (no <cls>/<eos>), mean over windows.
torch/transformers are imported lazily so clean_sequence() works without them."""
import os
import pickle
import threading

import numpy as np

import config

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
EXTRA_OK = set("XBUZO")                       # letters ESM's tokenizer knows


def clean_sequence(seq):
    """Accepts plain sequence or FASTA text.  Raises ValueError with a user-readable message."""
    if not isinstance(seq, str) or not seq.strip():
        raise ValueError("Protein sequence is empty.")
    lines = [l.strip() for l in seq.strip().splitlines() if l.strip() and not l.startswith(">")]
    s = "".join(lines).replace(" ", "").upper()
    if not s:
        raise ValueError("Protein sequence is empty.")
    if len(s) > config.MAX_SEQ_LEN:
        raise ValueError(f"Sequence is longer than {config.MAX_SEQ_LEN} residues.")
    bad = sorted(set(s) - STANDARD - EXTRA_OK)
    if bad:
        raise ValueError(f"Invalid amino-acid letters: {', '.join(bad)}")
    return s


class ESMEmbedder:
    """Lazy-loaded ESM-2 8M; one instance per process."""
    WINDOW, STRIDE, WIN_BATCH = 900, 800, 16

    def __init__(self, device=None):
        self._device = device
        self._tok = self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src = str(config.ESM_LOCAL_DIR) if config.ESM_LOCAL_DIR.exists() else config.ESM_NAME
        self._tok = AutoTokenizer.from_pretrained(src)
        self._model = AutoModel.from_pretrained(src).to(self.device).eval()

    def embed(self, seq):
        import torch
        with self._lock, torch.inference_mode():
            self._load()
            windows = [seq[k:k + self.WINDOW] for k in range(0, len(seq), self.STRIDE)]
            sums, count = np.zeros(320), 0
            for k in range(0, len(windows), self.WIN_BATCH):
                inp = self._tok(windows[k:k + self.WIN_BATCH], return_tensors="pt", padding=True,
                                truncation=True, max_length=1024).to(self.device)
                h = self._model(**inp).last_hidden_state
                mask = inp["attention_mask"].clone()
                last = mask.sum(1) - 1
                mask[:, 0] = 0                                                      # <cls>
                mask[torch.arange(len(last), device=self.device), last] = 0         # <eos>
                pooled = ((h * mask.unsqueeze(-1)).sum(1) /
                          mask.sum(1, keepdim=True).clamp(min=1)).float().cpu().numpy()
                sums += pooled.sum(0); count += len(pooled)
            return (sums / count).astype(np.float32)


class EmbeddingCache:
    """Training embeddings (read-only pickle from Drive) + new proteins computed by the app."""
    def __init__(self, embedder):
        self.embedder = embedder
        self._train = None
        self._user = None
        self._lock = threading.Lock()

    def _load(self):
        if self._train is None:
            self._train = pickle.load(open(config.TRAIN_EMB_PKL, "rb")) if config.TRAIN_EMB_PKL.exists() else {}
        if self._user is None:
            self._user = pickle.load(open(config.USER_EMB_PKL, "rb")) if config.USER_EMB_PKL.exists() else {}

    def known(self, seq):
        with self._lock:
            self._load()
            return seq in self._train or seq in self._user

    def get(self, seq):
        with self._lock:
            self._load()
            if seq in self._train:
                return self._train[seq]
            if seq in self._user:
                return self._user[seq]
        vec = self.embedder.embed(seq)                      # slow path (first time for this protein)
        with self._lock:
            self._user[seq] = vec
            config.CACHE_DIR.mkdir(exist_ok=True)
            tmp = str(config.USER_EMB_PKL) + ".tmp"
            pickle.dump(self._user, open(tmp, "wb"))
            os.replace(tmp, config.USER_EMB_PKL)
        return vec
