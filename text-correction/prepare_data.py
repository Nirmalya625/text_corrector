"""Download the Brown corpus (~1.1M words) and write data/corpus.txt."""
import io, os, urllib.request, zipfile

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus.txt")


def build(out=OUT):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
    lines = []
    for n in z.namelist():
        if n.startswith("brown/c") and not n.endswith("/"):
            for tok in z.read(n).decode("latin-1").split():
                lines.append(tok.rsplit("/", 1)[0] if "/" in tok else tok)
    with open(out, "w", encoding="utf-8") as f:
        f.write(" ".join(lines))
    return out


if __name__ == "__main__":
    print("wrote", build())
