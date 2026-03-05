import json
import zipfile
import sys


def main(path: str) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: inspect_cartridge.py <path.cartridge>")
        sys.exit(1)
    main(sys.argv[1])
