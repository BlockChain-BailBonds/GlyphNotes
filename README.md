# GlyphNotes
The Master Glyph Engine GGUF GLYPH
#!/usr/bin/env python3
import argparse, hashlib, json, os
from gguf import GGUFReader

ALPH = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # base32-ish, no 0/O/I/1 confusion

def short_id(name: str, n=3) -> str:
    h = hashlib.blake2s(name.encode('utf-8'), digest_size=4).digest()
    val = int.from_bytes(h, 'big')
    out = []
    for _ in range(n):
        out.append(ALPH[val & 31])
        val >>= 5
    return "".join(out)[::-1]

def main():
    ap = argparse.ArgumentParser(description="Build a glyph ledger for a GGUF model")
    ap.add_argument("--model", required=True, help="Path to .gguf")
    ap.add_argument("--out", default="glyph_ledger.json", help="Output ledger path")
    args = ap.parse_args()

    rd = GGUFReader(args.model)
    tensors = rd.tensors
    ledger = {"model": os.path.abspath(args.model), "tensors": []}

    seen_ids = set()
    for t in tensors:
        name = t.name
        sid = f"⟦{short_id(name)}⟧"
        # avoid collisions
        base = sid
        k = 1
        while sid in seen_ids:
            sid = f"{base[:-1]}{ALPH[k % len(ALPH)]}⟧"
            k += 1
        seen_ids.add(sid)
        entry = {
            "glyph": sid,
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.tensor_dtype.name if hasattr(t, 'tensor_dtype') else t.dtype),
        }
        ledger["tensors"].append(entry)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)

    print(f"[ledger] wrote {args.out} with {len(ledger['tensors'])} tensors")

if __name__ == "__main__":
    main()
