"""Resume the pinned public checkpoint; verify LFS hashes before evaluation."""

import concurrent.futures
import hashlib
import json
import pathlib
import subprocess


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    destination = root / ".models/Qwen2-Audio-7B-Instruct"
    url = "https://hf-mirror.com/api/models/Qwen/Qwen2-Audio-7B-Instruct?blobs=true"
    metadata = json.loads(
        subprocess.check_output(
            ["curl", "-f", "-L", "--max-time", "20", "-sS", url], text=True
        )
    )
    revision = "0a095220c30b7b31434169c3086508ef3ea5bf0a"
    assert metadata["sha"] == revision
    files = [
        item
        for item in metadata["siblings"]
        if item["rfilename"].endswith((".json", ".safetensors", ".txt"))
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
                    f"https://hf-mirror.com/Qwen/Qwen2-Audio-7B-Instruct/resolve/"
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
    print("All checkpoint files verified.", flush=True)


if __name__ == "__main__":
    main()
