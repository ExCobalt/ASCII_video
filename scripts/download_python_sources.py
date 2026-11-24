"""Download the Python source distributions pinned by uv.lock, checking SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def download(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    entries = []
    for package in lock["package"]:
        source = package.get("sdist")
        if source is None:
            if package["name"] != "ascii-video":
                raise ValueError(f"No source distribution for {package['name']}")
            continue
        url = source["url"]
        filename = Path(urllib.parse.urlparse(url).path).name
        target = output / filename
        expected = source["hash"].removeprefix("sha256:")
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch: {filename}")
            target.write_bytes(data)
        entries.append({"name": package["name"], "version": package["version"],
                        "filename": filename, "url": url, "sha256": expected})
        print(f"Verified {filename}", flush=True)
    (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/license-sources/python")
    download(parser.parse_args().output)
