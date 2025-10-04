import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # noqa: F401


DOMAIN_COLUMN_CANDIDATES = [
    "domain",
    "fqdn",
    "host",
    "dga_domain",
    "name",
    "url",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "class",
    "threat",
    "is_dga",
    "isDGA",
    "malicious",
]

DOMAIN_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\.]{0,252}$")


def is_probably_csv(sample: bytes) -> Tuple[bool, str]:
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"), delimiters=",\t;|")
        return True, dialect.delimiter
    except Exception:
        return False, ","


def normalize_domain(raw: str) -> Optional[str]:
    if not raw:
        return None
    d = raw.strip().lower().rstrip(".")
    if not d:
        return None
    # Filter obvious non-domain strings
    if "/" in d or " " in d:
        return None
    if not DOMAIN_REGEX.match(d):
        return None
    # Avoid pure TLDs or 1-char labels only
    if "." not in d:
        return None
    return d


def iter_text_file(path: Path, label: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support JSONL single-line objects
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    domain = obj.get("domain") or obj.get("fqdn") or obj.get("host")
                    if domain:
                        norm = normalize_domain(str(domain))
                        if norm:
                            lbl = obj.get("threat") or obj.get("label") or label
                            yield norm, str(lbl).lower()
                    continue
                except Exception:
                    pass
            norm = normalize_domain(line)
            if norm:
                yield norm, label


def iter_csv_file(path: Path, explicit_label: Optional[str]) -> Iterator[Tuple[str, str]]:
    with path.open("rb") as fb:
        head = fb.read(8192)
        fb.seek(0)
        is_csv, delimiter = is_probably_csv(head)
        text_stream: io.TextIOBase = io.TextIOWrapper(fb, encoding="utf-8", errors="ignore")
        reader = csv.DictReader(text_stream, delimiter=delimiter if is_csv else ",")
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        domain_key = next((k for k in fieldnames if k in DOMAIN_COLUMN_CANDIDATES), None)
        label_key = next((k for k in fieldnames if k in LABEL_COLUMN_CANDIDATES), None)
        if domain_key is None:
            # Fallback: use the first column if present
            if fieldnames:
                domain_key = fieldnames[0]
        for row in reader:
            if not row:
                continue
            raw_domain = row.get(domain_key) if domain_key else None
            if not raw_domain:
                continue
            norm = normalize_domain(str(raw_domain))
            if not norm:
                continue
            if explicit_label is not None:
                lbl = explicit_label
            else:
                raw_label = (row.get(label_key) if label_key else None) or ""
                raw_label = str(raw_label).strip().lower()
                if raw_label in ("dga", "1", "true", "malicious", "bad"):
                    lbl = "dga"
                elif raw_label in ("benign", "0", "false", "legit", "legitimate", "good", "normal"):
                    lbl = "benign"
                else:
                    # If unknown, skip row
                    continue
            yield norm, lbl


def detect_and_iter(path: Path, default_label: Optional[str]) -> Iterator[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        yield from iter_csv_file(path, default_label)
    else:
        yield from iter_text_file(path, default_label or "benign")


def walk_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.stat().st_size > 0:
            if p.suffix.lower() in {".csv", ".tsv", ".txt", ".list", ".data", ".jsonl"}:
                files.append(p)
    return files


essential_stdout = sys.stdout

def write_jsonl(records: Iterator[Tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for domain, lbl in records:
            lbl_norm = "dga" if lbl == "dga" else "benign"
            obj = {"domain": domain, "threat": lbl_norm}
            if orjson is not None:  # fast path
                out.write(orjson.dumps(obj).decode("utf-8"))
                out.write("\n")
            else:
                out.write(json.dumps(obj, ensure_ascii=False))
                out.write("\n")
            count += 1
            if count % 1_000_000 == 0:
                print(f"wrote {count:,} lines...", file=essential_stdout)
    print(f"done. total {count:,} lines written to {output_path}", file=essential_stdout)


def main():
    parser = argparse.ArgumentParser(description="Convert ExtraHop DGA dataset to JSONL {domain, threat}")
    parser.add_argument("--input_root", type=str, help="Root directory of the cloned ExtraHop dataset")
    parser.add_argument("--dga", type=str, nargs="*", help="Explicit DGA file(s)")
    parser.add_argument("--benign", type=str, nargs="*", help="Explicit benign file(s)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate domains (memory heavy for 10M+)")
    args = parser.parse_args()

    output_path = Path(args.output)

    to_process: List[Tuple[Path, Optional[str]]] = []

    if args.input_root:
        root = Path(args.input_root)
        all_files = walk_files(root)
        for f in all_files:
            name = f.name.lower()
            # Heuristics: filenames containing dga vs benign
            if any(tok in name for tok in ["dga", "malicious", "bot", "dgadomain"]):
                to_process.append((f, "dga"))
            elif any(tok in name for tok in ["benign", "legit", "alexa", "top", "whitelist"]):
                to_process.append((f, "benign"))
            else:
                # Unknown label from filename; let CSV label column decide
                to_process.append((f, None))

    if args.dga:
        for p in args.dga:
            to_process.append((Path(p), "dga"))
    if args.benign:
        for p in args.benign:
            to_process.append((Path(p), "benign"))

    if not to_process:
        raise SystemExit("No input files discovered. Provide --input_root or --dga/--benign files.")

    # Optionally deduplicate
    if args.dedup:
        seen = set()
        def generator() -> Iterator[Tuple[str, str]]:
            for fpath, label in to_process:
                for domain, lbl in detect_and_iter(fpath, label):
                    if domain in seen:
                        continue
                    seen.add(domain)
                    yield domain, lbl
        write_jsonl(generator(), output_path)
    else:
        def chain() -> Iterator[Tuple[str, str]]:
            for fpath, label in to_process:
                yield from detect_and_iter(fpath, label)
        write_jsonl(chain(), output_path)


if __name__ == "__main__":
    main()
