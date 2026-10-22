"""Shared machinery for the reference-data generators.

``generate_d3_data.py`` and ``generate_d4_data.py`` both read exact upstream
files pinned by commit and sha256, parse Fortran data statements strictly,
pack the results into a sectioned little-endian container, and refuse to write
anything that does not regenerate byte for byte. The pieces they share live
here: source pinning and download, the Fortran lexer, mctc-lib's unit
constants, the binary container writer, deterministic JSON, and the
command-line driver.

The generators run as ``python -I tools/...``, which leaves this directory off
``sys.path``; they insert it explicitly before importing this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "src" / "vanderfast" / "data"

HEADER_BYTES = 512
SECTION_ALIGN = 64
BYTE_ORDER_MARK = 0x01020304
SOURCE_ID_OFFSET = 48
SOURCE_ID_BYTES = 64
SECTION_COUNT_OFFSET = 112
SECTION_TABLE_OFFSET = 120
SECTION_ENTRY = struct.Struct("<IIQQ")
MAX_SECTIONS = (HEADER_BYTES - SECTION_TABLE_OFFSET) // SECTION_ENTRY.size

DTYPE_U8 = 1
DTYPE_F64 = 2


class GenerationError(RuntimeError):
    """Raised when an upstream source does not match the strict expectations."""


# --------------------------------------------------------------------------
# Source access
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePin:
    """One pinned upstream repository."""

    name: str
    repository: str
    commit: str
    files: dict[str, str]
    meta: dict[str, Any]


def load_pins(path: Path) -> dict[str, SourcePin]:
    """Read pinned sources from a ``vanderfast.sources/1`` TOML file."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    pins: dict[str, SourcePin] = {}
    for name, entry in document["sources"].items():
        meta = {k: v for k, v in entry.items() if k != "files"}
        pins[name] = SourcePin(
            name=name,
            repository=entry["repository"],
            commit=entry["commit"],
            files=dict(entry["files"]),
            meta=meta,
        )
    return pins


def _raw_url(pin: SourcePin, relpath: str) -> str:
    owner_repo = pin.repository.removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{pin.commit}/{relpath}"


def default_cache() -> Path:
    """Return the download cache, keyed by sha256 so sources share entries."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vanderfast" / "upstream"


class SourceReader:
    """Return verified bytes of pinned upstream files."""

    def __init__(
        self,
        pins: dict[str, SourcePin],
        source_dirs: dict[str, Path],
        cache: Path,
    ) -> None:
        """Create a reader using local checkouts first, then a download cache."""
        self._pins = pins
        self._source_dirs = source_dirs
        self._cache = cache

    def read(self, source: str, relpath: str) -> bytes:
        """Return the bytes of ``relpath`` after verifying its pinned sha256."""
        pin = self._pins[source]
        expected = pin.files[relpath]
        if source in self._source_dirs:
            data = (self._source_dirs[source] / relpath).read_bytes()
        else:
            cached = self._cache / expected
            if cached.is_file():
                data = cached.read_bytes()
            else:
                url = _raw_url(pin, relpath)
                if not url.startswith("https://raw.githubusercontent.com/"):
                    msg = f"refusing to fetch non-GitHub URL {url}"
                    raise GenerationError(msg)
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            msg = f"{source}:{relpath} sha256 {digest} does not match pinned {expected}"
            raise GenerationError(msg)
        if source not in self._source_dirs:
            self._cache.mkdir(parents=True, exist_ok=True)
            (self._cache / expected).write_bytes(data)
        return data

    def text(self, source: str, relpath: str) -> str:
        """Return a verified file decoded as UTF-8."""
        return self.read(source, relpath).decode("utf-8")


# --------------------------------------------------------------------------
# Fortran lexing
# --------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Remove a trailing ``!`` comment that is not inside a string literal."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "!":
            return line[:index]
    return line


def _split_semicolons(statement: str) -> list[str]:
    parts: list[str] = []
    quote: str | None = None
    start = 0
    for index, char in enumerate(statement):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == ";":
            parts.append(statement[start:index])
            start = index + 1
    parts.append(statement[start:])
    return [part.strip() for part in parts if part.strip()]


def fortran_statements(source: str) -> list[str]:
    """Split free-form Fortran into comment-free, continuation-joined statements."""
    statements: list[str] = []
    pending = ""
    for raw in source.splitlines():
        line = _strip_comment(raw).rstrip()
        stripped = line.lstrip()
        if pending and not stripped:
            continue  # comment or blank lines may sit between continuation lines
        if pending and stripped.startswith("&"):
            stripped = stripped[1:]
        if not pending:
            stripped = line.strip()
        if stripped.endswith("&"):
            pending += stripped[:-1] + " "
            continue
        full = (pending + stripped).strip()
        pending = ""
        if full:
            statements.extend(_split_semicolons(full))
    if pending.strip():
        msg = "unterminated continuation at end of file"
        raise GenerationError(msg)
    return [re.sub(r"\s+", " ", statement) for statement in statements]


def squash(text: str) -> str:
    """Remove all whitespace (for comparing statements to pinned source lines)."""
    return re.sub(r"\s+", "", text)


REAL_LITERAL = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eEdD][+-]?\d+)?_wp")
INT_LITERAL = re.compile(r"[+-]?\d+")


def parse_real_list(body: str) -> list[float]:
    """Parse a comma-separated list of ``_wp`` real literals strictly."""
    values: list[float] = []
    for token in (part.strip() for part in body.split(",")):
        if not REAL_LITERAL.fullmatch(token):
            msg = f"unexpected token in real array: {token!r}"
            raise GenerationError(msg)
        mantissa = token.removesuffix("_wp").replace("d", "e").replace("D", "e")
        values.append(float(mantissa))
    return values


def parse_int_list(body: str) -> list[int]:
    """Parse a comma-separated list of integer literals strictly."""
    values: list[int] = []
    for token in (part.strip() for part in body.split(",")):
        if not INT_LITERAL.fullmatch(token):
            msg = f"unexpected token in integer array: {token!r}"
            raise GenerationError(msg)
        values.append(int(token))
    return values


def find_one(statements: Iterable[str], pattern: str) -> re.Match[str]:
    """Return the single statement that fully matches ``pattern``."""
    regex = re.compile(pattern)
    matches = [m for m in (regex.fullmatch(s) for s in statements) if m]
    if len(matches) != 1:
        msg = f"expected exactly one statement matching {pattern!r}, found {len(matches)}"
        raise GenerationError(msg)
    return matches[0]


def int_parameter(statements: Sequence[str], name: str) -> int:
    """Return the value of ``integer, parameter :: name = N``."""
    match = find_one(statements, rf"integer, parameter :: {name} = (\d+)")
    return int(match.group(1))


def check_convention_lines(
    reader: SourceReader, table: Mapping[tuple[str, str], Sequence[tuple[str, str]]]
) -> dict[str, str]:
    """Assert convention-bearing source lines and return ``key -> "relpath: line"``.

    ``table`` maps ``(source, relpath)`` to ``(key, line)`` pairs. Lines are
    compared after removing all whitespace, against the file's statements.
    """
    found: dict[str, str] = {}
    for (source, relpath), lines in table.items():
        statements = {squash(s) for s in fortran_statements(reader.text(source, relpath))}
        joined = "\n".join(statements)
        for key, line in lines:
            if squash(line) not in joined:
                msg = f"{relpath}: convention {key} not found: {line!r}"
                raise GenerationError(msg)
            if key in found:
                msg = f"duplicate convention key {key}"
                raise GenerationError(msg)
            found[key] = f"{relpath}: {line}"
    return found


# --------------------------------------------------------------------------
# mctc-lib unit constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Constants:
    """Unit constants evaluated exactly as mctc-lib evaluates them."""

    aatoau: float
    autoaa: float
    pi: float
    codata: dict[str, float]


def parse_constants(reader: SourceReader, source: str = "mctc-lib") -> Constants:
    """Evaluate mctc-lib's ``aatoau`` from its CODATA 2018 definition."""
    codata_statements = fortran_statements(reader.text(source, "src/mctc/io/codata2018.f90"))
    wanted = {
        "planck_constant": "h",
        "speed_of_light_in_vacuum": "c",
        "fine_structure_constant": "alpha",
        "electron_mass": "me",
    }
    codata: dict[str, float] = {}
    pattern = re.compile(
        r"real\(wp\), parameter, public :: (\w+) = (" + REAL_LITERAL.pattern + ")",
    )
    for statement in codata_statements:
        match = pattern.fullmatch(statement)
        if match and match.group(1).lower() in wanted:
            key = wanted[match.group(1).lower()]
            if key in codata:
                msg = f"duplicate CODATA constant {match.group(1)}"
                raise GenerationError(msg)
            codata[key] = parse_real_list(match.group(2))[0]
    if set(codata) != set(wanted.values()):
        msg = f"missing CODATA constants: {set(wanted.values()) - set(codata)}"
        raise GenerationError(msg)

    constant_statements = fortran_statements(reader.text(source, "src/mctc/io/constants.f90"))
    pi_match = find_one(
        constant_statements,
        r"real\(wp\), parameter :: pi = (" + REAL_LITERAL.pattern + ")",
    )
    pi = parse_real_list(pi_match.group(1))[0]
    for field, name in [
        ("h", "planck_constant"),
        ("c", "speed_of_light_in_vacuum"),
        ("alpha", "fine_structure_constant"),
        ("me", "electron_mass"),
    ]:
        find_one(constant_statements, rf"real\(wp\) :: {field} = {name}")

    convert = {
        squash(s) for s in fortran_statements(reader.text(source, "src/mctc/io/convert.f90"))
    }
    required = [
        "real(wp),parameter::hbar=codata%h/(2.0_wp*pi)",
        "real(wp),parameter::bohr=hbar/(codata%me*codata%c*codata%alpha)",
        "real(wp),public,parameter::autoaa=bohr*1e10_wp",
        "real(wp),public,parameter::aatoau=1.0_wp/autoaa",
    ]
    for line in required:
        if line not in convert:
            msg = f"mctc convert.f90 no longer contains {line!r}"
            raise GenerationError(msg)

    hbar = codata["h"] / (2.0 * pi)
    bohr = hbar / (codata["me"] * codata["c"] * codata["alpha"])
    autoaa = bohr * 1e10
    aatoau = 1.0 / autoaa
    return Constants(aatoau=aatoau, autoaa=autoaa, pi=pi, codata=codata)


# --------------------------------------------------------------------------
# Binary container
# --------------------------------------------------------------------------


def pack_f64(values: Sequence[float]) -> bytes:
    """Pack doubles little-endian."""
    return struct.pack(f"<{len(values)}d", *values)


@dataclass(frozen=True)
class Section:
    """One typed, 64-byte aligned payload of a reference-data container."""

    section_id: int
    dtype: int
    payload: bytes
    count: int
    name: str


def build_container(
    prefix: bytes, source_id: str, sections: Sequence[Section]
) -> tuple[bytes, dict[str, str]]:
    """Assemble header, section table, and aligned payloads; return per-section hashes.

    ``prefix`` fills header bytes ``[0, 48)`` (magic, version, header size,
    byte-order mark, flags, and format-specific counts). The source id occupies
    bytes ``[48, 112)``, the section count byte 112, and the section table
    (``<IIQQ``: id, dtype, offset, count) starts at byte 120.
    """
    if len(prefix) != SOURCE_ID_OFFSET:
        msg = f"header prefix must be {SOURCE_ID_OFFSET} bytes, got {len(prefix)}"
        raise GenerationError(msg)
    if len(sections) > MAX_SECTIONS:
        msg = f"at most {MAX_SECTIONS} sections fit in the header"
        raise GenerationError(msg)
    body = bytearray()
    table = bytearray()
    offset = HEADER_BYTES
    hashes: dict[str, str] = {}
    for section in sections:
        table += SECTION_ENTRY.pack(section.section_id, section.dtype, offset, section.count)
        body += section.payload
        offset += len(section.payload)
        pad = (-offset) % SECTION_ALIGN
        body += b"\0" * pad
        offset += pad
        hashes[section.name] = hashlib.sha256(section.payload).hexdigest()

    source_bytes = source_id.encode("utf-8")
    if len(source_bytes) >= SOURCE_ID_BYTES:
        msg = "source id too long"
        raise GenerationError(msg)
    header = bytearray(HEADER_BYTES)
    header[:SOURCE_ID_OFFSET] = prefix
    header[SOURCE_ID_OFFSET : SOURCE_ID_OFFSET + len(source_bytes)] = source_bytes
    struct.pack_into("<II", header, SECTION_COUNT_OFFSET, len(sections), 0)
    header[SECTION_TABLE_OFFSET : SECTION_TABLE_OFFSET + len(table)] = table
    return bytes(header) + bytes(body), hashes


# --------------------------------------------------------------------------
# JSON and provenance
# --------------------------------------------------------------------------


def dumps(document: Any) -> bytes:
    """Serialise JSON deterministically."""
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def manifest_sources(pins: Mapping[str, SourcePin]) -> list[dict[str, Any]]:
    """Return the manifest's ``sources`` list (pin metadata plus file hashes)."""
    return [
        {
            "name": pin.name,
            **{k: v for k, v in pin.meta.items() if k != "repository"},
            "repository": pin.repository,
            "files": dict(sorted(pin.files.items())),
        }
        for pin in pins.values()
    ]


def manifest_outputs(outputs: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    """Return the manifest's ``outputs`` map (sha256 and size per file)."""
    return {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in sorted(outputs.items())
    }


def check_licence_copies(reader: SourceReader, pins: Mapping[str, SourcePin]) -> None:
    """Require ``data/licenses/<source>-<file>`` to equal each pinned licence file."""
    for pin in pins.values():
        for relpath in pin.meta["license_files"]:
            copy = REPO_ROOT / "data" / "licenses" / f"{pin.name}-{relpath}"
            if not copy.is_file() or copy.read_bytes() != reader.read(pin.name, relpath):
                msg = f"{copy} must be a verbatim copy of {pin.name}:{relpath}"
                raise GenerationError(msg)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_source_dirs(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``--source-dir NAME=DIR`` options."""
    result: dict[str, Path] = {}
    for value in values:
        name, sep, directory = value.partition("=")
        if not sep:
            msg = f"--source-dir expects NAME=DIR, got {value!r}"
            raise SystemExit(msg)
        result[name] = Path(directory).expanduser()
    return result


@dataclass(frozen=True)
class Generated:
    """A generator run: output bytes by file name plus a binary-verification hook."""

    outputs: dict[str, bytes]
    verify_binary: Callable[[Path], list[str]]


def run_cli(
    argv: Sequence[str] | None,
    *,
    description: str,
    script: str,
    sources_toml: Path,
    generate: Callable[[SourceReader, dict[str, SourcePin]], Generated],
    verify_help: str,
) -> int:
    """Parse the common generator options, generate, then check, verify, or write."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="read a pinned source from a local checkout (still hash-checked)",
    )
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="require committed outputs to be byte-identical to a fresh generation",
    )
    parser.add_argument("--verify-binary", type=Path, metavar="LIBDIR", help=verify_help)
    args = parser.parse_args(argv)

    pins = load_pins(sources_toml)
    reader = SourceReader(pins, parse_source_dirs(args.source_dir), args.cache)
    generated = generate(reader, pins)

    status = 0
    if args.verify_binary is not None:
        failures = generated.verify_binary(args.verify_binary)
        if failures:
            print("binary verification failed for: " + ", ".join(failures), file=sys.stderr)
            status = 1
        else:
            print("binary verification: all arrays found in the compiled reference")

    if args.check:
        for name, data in generated.outputs.items():
            path = args.output_dir / name
            if not path.is_file() or path.read_bytes() != data:
                print(f"{path} is out of date; run {script}", file=sys.stderr)
                status = 1
        if status == 0:
            print("generated data is up to date")
        return status

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in generated.outputs.items():
        (args.output_dir / name).write_bytes(data)
        print(f"wrote {args.output_dir / name} ({len(data)} bytes)")
    return status
