"""Clone torchMoji at a pinned commit, apply the Python 3 / PyTorch 2 fixes, download the weights.

    python -m deepmoji.setup_torchmoji

Everything goes to third_party/torchMoji (git-ignored). Safe to re-run: finished steps are skipped.

torchMoji (https://github.com/huggingface/torchMoji, MIT license) is the PyTorch port of DeepMoji
(Felbo et al., 2017). Its last commit predates Python 3.11 and PyTorch 1.1, and three things break
or silently change on current versions. Each fix below restores the original behaviour; after
patching, torchMoji's own reference outputs are reproduced exactly (python -m deepmoji.check_torchmoji).
"""

import hashlib
import subprocess
import sys
import urllib.request

from deepmoji.common import TORCHMOJI_DIR

REPO = "https://github.com/huggingface/torchMoji.git"
COMMIT = "9dd01df645dd84b3645633f55067c0bbcac90011"  # latest master, checked 2026-11-15
WEIGHTS_URL = "https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=1"  # from scripts/download_weights.py
WEIGHTS_SHA256 = "8cbf6f7067d56aa1c2d571bb169f05fba16cea4c263c06fb3f217f42c591a978"  # 89,616,062 bytes

# (file, original text, replacement). Each original must occur exactly once.
PATCHES = [
    # 1. A mid-pattern inline flag "(?i)" is an error since Python 3.11. Old Python applied it to the
    #    whole pattern, so drop it here and compile the whole pattern with IGNORECASE instead.
    ("torchmoji/tokenizer.py",
     "RE_TITLES = r'|'.join([r'(?i)' + t for t in TITLES])",
     "RE_TITLES = r'|'.join(TITLES)  # patched (CIS5300): IGNORECASE is set on RE_PATTERN below"),
    ("torchmoji/tokenizer.py",
     "                        re.UNICODE)",
     "                        re.UNICODE | re.IGNORECASE)  # patched (CIS5300): was a mid-pattern (?i)"),
    # 2. PackedSequence has four fields since PyTorch 1.1, so unpacking it into two fails.
    ("torchmoji/lstm.py",
     "            input, batch_sizes = input\n",
     "            input, batch_sizes = input.data, input.batch_sizes  # patched (CIS5300): 4 fields since torch 1.1\n"),
    # 3. Python 3 str has no .decode, so these "is it ASCII?" checks always failed. No error is raised,
    #    but shorten_word() then never shortens "!!!" / "..." / "ewww" to their in-vocabulary forms
    #    "!!" / ".." / "eww", which turns them into unknown words and changes the features.
    ("torchmoji/filter_utils.py",
     "        word.decode('ascii')\n",
     "        word.encode('ascii')  # patched (CIS5300): str has no .decode on Python 3\n"),
    ("torchmoji/word_generator.py",
     "            word.decode('ascii')\n",
     "            word.encode('ascii')  # patched (CIS5300): str has no .decode on Python 3\n"),
]


def run(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def clone():
    if (TORCHMOJI_DIR / ".git").is_dir():
        print(f"{TORCHMOJI_DIR} exists, skipping clone")
    else:
        TORCHMOJI_DIR.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", REPO, str(TORCHMOJI_DIR))
    head = subprocess.run(["git", "-C", str(TORCHMOJI_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        run("git", "-C", str(TORCHMOJI_DIR), "checkout", "--quiet", COMMIT)


def patch():
    for name, old, new in PATCHES:
        path = TORCHMOJI_DIR / name
        # newline="" keeps the file's own line endings (CRLF on Windows checkouts with autocrlf).
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        crlf = "\r\n" in text
        old_, new_ = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
        if new_ in text:
            print(f"already patched: {name}: {new.strip()[:60]}")
            continue
        if text.count(old_) != 1:
            sys.exit(f"cannot patch {name}: expected exactly one occurrence of {old.strip()!r}")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text.replace(old_, new_))
        print(f"patched {name}: {new.strip()[:60]}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights():
    path = TORCHMOJI_DIR / "model" / "pytorch_model.bin"
    if path.exists() and sha256(path) == WEIGHTS_SHA256:
        print(f"weights present and verified: {path}")
        return
    print(f"downloading pretrained weights (about 85 MB) from {WEIGHTS_URL}", flush=True)
    urllib.request.urlretrieve(WEIGHTS_URL, path)
    actual = sha256(path)
    if actual != WEIGHTS_SHA256:
        path.unlink()
        sys.exit(f"weights checksum mismatch: got {actual}, expected {WEIGHTS_SHA256}")
    print(f"weights downloaded and verified: {path}")


def main():
    clone()
    patch()
    download_weights()
    print("done. next: python -m deepmoji.check_torchmoji")


if __name__ == "__main__":
    main()
