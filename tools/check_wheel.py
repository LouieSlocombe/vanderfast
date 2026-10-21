"""Check a CUDA wheel: device code for every architecture and its dynamic dependencies."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ALLOWED_NEEDED = re.compile(
    r"^(libstdc\+\+\.so\.6|libgcc_s\.so\.1|libm\.so\.6|libc\.so\.6|libdl\.so\.2|"
    r"librt\.so\.1|libpthread\.so\.0|ld-linux-x86-64\.so\.2)$"
)


def _run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603


def check(wheel: Path, architectures: list[int], ptx: int | None) -> list[str]:
    """Return a list of problems found in ``wheel``."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        libraries = list(Path(directory).rglob("_native*.so"))
        if len(libraries) != 1:
            return [f"expected one _native extension, found {len(libraries)}"]
        library = str(libraries[0])
        elf = _run("cuobjdump", "--list-elf", library)
        found = {int(m) for m in re.findall(r"sm_(\d+)", elf)}
        problems += [f"missing SASS for sm_{a}" for a in architectures if a not in found]
        if ptx is not None:
            listed = _run("cuobjdump", "--list-ptx", library)
            if f"sm_{ptx}" not in listed and f"compute_{ptx}" not in listed:
                problems.append(f"missing PTX for compute_{ptx}")
        readelf = shutil.which("readelf") or "readelf"
        needed = re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", _run(readelf, "-d", library))
        problems += [
            f"unexpected dynamic dependency {n}" for n in needed if not ALLOWED_NEEDED.match(n)
        ]
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sass", default="75,80,86,89,90,100,120")
    parser.add_argument("--ptx", type=int, default=120)
    args = parser.parse_args(argv)
    problems = check(args.wheel, [int(a) for a in args.sass.split(",") if a], args.ptx)
    for problem in problems:
        print(f"{args.wheel}: {problem}", file=sys.stderr)
    if not problems:
        print(f"{args.wheel}: device code and dependencies OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
