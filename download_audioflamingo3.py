"""Download only the pinned base checkpoint; verify every LFS SHA-256."""

import concurrent.futures
import hashlib
import json
import pathlib
import subprocess


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    destination = root / ".models/audio-flamingo-3-hf"
    destination.mkdir(exist_ok=True)
    metadata = json.loads(
        pathlib.Path(__file__).with_name("audioflamingo3-hub.json").read_text()
    )
    revision = "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
    assert metadata["sha"] == revision
    files = [
        item
        for item in metadata["siblings"]
        if "/" not in item["rfilename"]
        and item["rfilename"] != "model.safetensors"
        and item["rfilename"].endswith((".json", ".safetensors", ".txt", ".jinja"))
    ]

    def download(item):
        name = item["rfilename"]
        target = destination / name
        if not target.exists() or target.stat().st_size != item["size"]:
            result = subprocess.run(
                [
                    "curl",
                    "-f",
                    "-L",
                    "--retry",
                    "2",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "1800",
                    "--continue-at",
                    "-",
                    "-sS",
                    f"https://hf-mirror.com/nvidia/audio-flamingo-3-hf/resolve/"
                    f"{revision}/{name}?download=true",
                    "-o",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                print(name, result.returncode, result.stderr[:300], flush=True)
                return False
        assert target.stat().st_size == item["size"], name
        if "lfs" in item:
            digest = hashlib.sha256()
            with target.open("rb") as file:
                while block := file.read(8 * 1024 * 1024):
                    digest.update(block)
            assert digest.hexdigest() == item["lfs"]["sha256"], name
        print(name, "verified", target.stat().st_size, flush=True)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(download, files))
    assert all(results), "Some downloads failed; rerun to resume."
    (destination / "source-revision.txt").write_text(revision + "\n")
    print("All base checkpoint files verified.", flush=True)


if __name__ == "__main__":
    main()
