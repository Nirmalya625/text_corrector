"""Context-aware spell checker + light grammar fixer.

Pipeline: tokenise -> spelling (edit-distance candidates ranked by unigram
prior + bigram context + error-model penalty) -> real-word confusables
(their/there...) -> grammar rules (a/an, repeats, "i", capitalisation).
"""
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "corpus.txt")
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+")
LETTERS = "abcdefghijklmnopqrstuvwxyz'"
CONFUSABLES = [
    {"their", "there", "they're"}, {"your", "you're"}, {"its", "it's"}, {"then", "than"},
    {"to", "too"}, {"affect", "effect"}, {"loose", "lose"}, {"whose", "who's"},
    {"weather", "whether"}, {"quiet", "quite"}, {"principal", "principle"},
]
CONF_MAP = {w: s for s in CONFUSABLES for w in s}
FALLBACK = ("the quick brown fox jumps over the lazy dog and the dog was very "
            "happy to see the fox in the house because they were friends " * 20)


def _match_case(src, dst):
    if len(src) > 1 and src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


class SpellCorrector:
    def __init__(self, corpus_path=DATA, min_count=2):
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", self._load(corpus_path).lower())
        self.uni = Counter(words)
        self.bi = Counter(zip(words, words[1:]))
        self.N, self.V, self.min_count = len(words), len(self.uni), min_count

    @staticmethod
    def _load(path):
        if not os.path.exists(path):
            try:
                from prepare_data import build
                build(path)
            except Exception:
                return FALLBACK
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ---- language model -------------------------------------------------
    def known(self, w):
        return self.uni.get(w, 0) >= self.min_count

    def lp_uni(self, w):
        return math.log((self.uni.get(w, 0) + 0.5) / (self.N + self.V))

    def lp_bi(self, a, b):
        return math.log((self.bi.get((a, b), 0) + 0.01) / (self.uni.get(a, 0) + 0.01 * self.V))

    def context(self, prev, w, nxt):
        s = 0.0
        if prev: s += self.lp_bi(prev, w)
        if nxt: s += self.lp_bi(w, nxt)
        return s

    # ---- candidate generation ------------------------------------------
    @staticmethod
    def edits1(w):
        sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        dele = [a + b[1:] for a, b in sp if b]
        trans = [a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1]
        rep = [a + c + b[1:] for a, b in sp if b for c in LETTERS]
        ins = [a + c + b for a, b in sp for c in LETTERS]
        return set(dele + trans + rep + ins)

    def candidates(self, w):
        e1 = self.edits1(w)
        c1 = {x for x in e1 if self.known(x)}
        if c1:
            return {x: 1 for x in c1}
        c2 = {y for x in e1 for y in self.edits1(x) if self.known(y)}
        return {x: 2 for x in c2} or {w: 0}

    def rank(self, w, prev=None, nxt=None, k=5):
        cands = self.candidates(w)
        scored = {c: self.lp_uni(c) + 0.8 * self.context(prev, c, nxt) - 3.0 * d
                  for c, d in cands.items()}
        return sorted(scored, key=scored.get, reverse=True)[:k]

    def suggest(self, word, k=5):
        w = word.lower()
        return [_match_case(word, s) for s in self.rank(w, k=k)] if not self.known(w) else [word]

    # ---- main API -------------------------------------------------------
    def correct(self, text):
        toks = [{"t": m.group(), "s": m.start(), "w": bool(re.match(r"[A-Za-z]", m.group()))}
                for m in TOKEN.finditer(text)]
        widx = [i for i, t in enumerate(toks) if t["w"]]
        changes = []

        def neighbour(pos, step):
            j = pos + step
            if not 0 <= j < len(widx):
                return None
            gap = toks[widx[j]]["s"] - (toks[widx[pos]]["s"] + len(toks[widx[pos]]["t"])) if step > 0 else \
                  toks[widx[pos]]["s"] - (toks[widx[j]]["s"] + len(toks[widx[j]]["t"]))
            if gap > 1 or (gap == 1 and not text[min(toks[widx[pos]]["s"], toks[widx[j]]["s"]) +
                            len(toks[widx[min(pos, j)]]["t"])].isspace()):
                return None
            return toks[widx[j]]["out"].lower() if step < 0 else toks[widx[j]]["t"].lower()

        def log(tok, new, kind, sugg=None):
            changes.append({"original": tok["t"], "corrected": new, "type": kind,
                            "start": tok["s"], "end": tok["s"] + len(tok["t"]),
                            "suggestions": sugg or [new]})
            tok["out"] = new

        for p, i in enumerate(widx):
            tok = toks[i]
            tok["out"] = tok["t"]
            w = tok["t"].lower()
            prev, nxt = neighbour(p, -1) if p else None, neighbour(p, 1)
            sentence_start = p == 0 or re.search(r"[.!?]\s*$", "".join(t["t"] for t in toks[widx[p-1]+1:i]))
            if tok["t"][0].isupper() and not sentence_start and not tok["t"].isupper():
                continue                                  # likely a proper noun
            if not self.known(w):                         # 1) non-word spelling error
                ranked = self.rank(w, prev, nxt)
                if ranked and ranked[0] != w:
                    log(tok, _match_case(tok["t"], ranked[0]), "spelling",
                        [_match_case(tok["t"], r) for r in ranked])
            elif w in CONF_MAP:                           # 2) real-word confusable
                alts = CONF_MAP[w]
                best = max(alts, key=lambda a: self.context(prev, a, nxt))
                if best != w and self.context(prev, best, nxt) - self.context(prev, w, nxt) > 4:
                    log(tok, _match_case(tok["t"], best), "context", sorted(alts))

        # 3) grammar rules on the corrected word stream
        drop = set()
        for p, i in enumerate(widx):
            tok, w = toks[i], toks[i]["out"]
            nxt = toks[widx[p + 1]]["out"].lower() if p + 1 < len(widx) else ""
            prv = toks[widx[p - 1]]["out"].lower() if p else ""
            if p and w.lower() == prv and w.lower() not in {"had", "that", "very"}:
                drop.add(i); log(tok, "", "repeated word"); continue
            if w.lower() == "i":
                if w != "I": log(tok, "I", "capitalization")
            elif w.lower() in ("a", "an") and nxt:
                vowel = nxt[0] in "aeiou" and not nxt.startswith(("uni", "use", "one", "eu"))
                vowel = vowel or nxt.startswith(("hour", "honest", "heir"))
                want = _match_case(w, "an" if vowel else "a")
                if want != w: log(tok, want, "a/an")

        out, cap = [], True
        for i, t in enumerate(toks):
            if i in drop:
                if out and out[-1].isspace(): out.pop()
                continue
            s = t.get("out", t["t"]) if t["w"] else t["t"]
            if t["w"]:
                if cap and s[:1].islower():
                    new = s[:1].upper() + s[1:]
                    changes.append({"original": s, "corrected": new, "type": "capitalization",
                                    "start": t["s"], "end": t["s"] + len(t["t"]), "suggestions": [new]})
                    s = new
                cap = False
            elif re.search(r"[.!?]\s*$|\n\s*$", s):
                cap = True
            out.append(s)
        corrected = re.sub(r"[ \t]+([,.;:!?])", r"\1", "".join(out))
        corrected = re.sub(r"[ \t]{2,}", " ", corrected)
        # merge duplicate changes on the same span (spelling + capitalization)
        merged = {}
        for c in changes:
            merged[(c["start"], c["end"])] = {**merged.get((c["start"], c["end"]), {}), **c}
        return {"original": text, "corrected": corrected,
                "changes": sorted(merged.values(), key=lambda c: c["start"])}


# ---- evaluation -------------------------------------------------------------
TEST_PAIRS = [
    ("speling", "spelling"), ("korrect", "correct"), ("recieve", "receive"),
    ("definately", "definitely"), ("seperate", "separate"), ("occured", "occurred"),
    ("wich", "which"), ("beleive", "believe"), ("tommorow", "tomorrow"),
    ("necesary", "necessary"), ("freind", "friend"), ("goverment", "government"),
    ("adress", "address"), ("untill", "until"), ("becuase", "because"),
    ("thier", "their"), ("acheive", "achieve"), ("enviroment", "environment"),
    ("peice", "piece"), ("langauge", "language"), ("importent", "important"),
    ("writting", "writing"), ("diffrent", "different"), ("beutiful", "beautiful"),
    ("knowlege", "knowledge"), ("sucess", "success"), ("prefered", "preferred"),
    ("tomorow", "tomorrow"), ("realy", "really"), ("accomodate", "accommodate"),
]


def evaluate(sc, pairs=TEST_PAIRS):
    hits = [(w, sc.suggest(w, 1)[0], t) for w, t in pairs]
    acc = sum(g == t for _, g, t in hits) / len(hits)
    return acc, [h for h in hits if h[1] != h[2]]


if __name__ == "__main__":
    sc = SpellCorrector()
    acc, miss = evaluate(sc)
    print(f"top-1 accuracy: {acc:.1%}  misses: {miss}")
    print(sc.correct("i went to a apple store and bought an new phone. Their going to the the park tomorow")["corrected"])
