"""Verify the packaged release without loading model files or fitting models."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
failures = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("Missing or changed files:\n" + "\n".join(failures))
print(f"Verified {len(manifest)} release files.")
