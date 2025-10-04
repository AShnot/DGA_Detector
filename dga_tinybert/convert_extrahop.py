import argparse
import json
import os
import re
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, track

console = Console()

VALID_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+$")


def is_valid_domain(value: str) -> bool:
    if not value:
        return False
    value = value.strip().lower()
    if len(value) < 2 or len(value) > 253:
        return False
    if not VALID_DOMAIN_RE.match(value):
        return False
    # no label longer than 63 and no empty labels
    labels = value.split(".")
    if any(len(lbl) == 0 or len(lbl) > 63 for lbl in labels):
        return False
    return True


def normalize_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    r = str(raw).strip().lower()
    if r in {"benign", "legit", "legitimate", "normal", "clean", "good", "non_dga", "nondga", "false", "0"}:
        return "benign"
    if r in {"dga", "malicious", "generated", "true", "1"}:
        return "dga"
    # Some datasets provide DGA family names; treat any non-benign label as dga
    return "dga"


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Returns (domain_col, label_col, bool_col)
    lower_cols = {c.lower(): c for c in df.columns}
    domain_candidates = [
        "domain", "fqdn", "hostname", "host", "qname", "dns_qname",
    ]
    label_candidates = [
        "label", "type", "tag", "class", "threat", "category", "dga_family", "family"
    ]
    bool_candidates = ["is_dga", "is-dga", "dga", "malicious", "is_malicious"]

    domain_col = next((lower_cols[c] for c in domain_candidates if c in lower_cols), None)
    label_col = next((lower_cols[c] for c in label_candidates if c in lower_cols), None)
    bool_col = next((lower_cols[c] for c in bool_candidates if c in lower_cols), None)
    return domain_col, label_col, bool_col


def iter_csv_records(csv_path: str, chunksize: int = 200_000) -> Iterable[Tuple[str, str]]:
    # Try common separators automatically
    seps = [",", "\t", ";"]
    read_ok = False
    last_err = None
    for sep in seps:
        try:
            for chunk in pd.read_csv(
                csv_path,
                sep=sep,
                dtype=str,
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines="skip",
                engine="python" if sep != "," else "c",
            ):
                domain_col, label_col, bool_col = detect_columns(chunk)
                if domain_col is None:
                    # Try to infer from URL
                    if "url" in {c.lower() for c in chunk.columns}:
                        df = chunk.rename(columns={next(c for c in chunk.columns if c.lower()=="url"): "url"})
                        # Extract host from URL
                        hosts = df["url"].astype(str).str.extract(r"^[a-zA-Z]+://([^/]+)")[0].str.lower()
                        domains = hosts
                    else:
                        # No domain column; skip
                        continue
                else:
                    domains = chunk[domain_col].astype(str).str.lower().str.strip()

                # Determine labels
                if bool_col is not None and bool_col != domain_col:
                    raw = chunk[bool_col]
                    labels = raw.apply(lambda x: normalize_label(str(x)))
                elif label_col is not None and label_col != domain_col:
                    raw = chunk[label_col]
                    labels = raw.apply(normalize_label)
                else:
                    # If there is a dga_family or similar, treat as dga
                    labels = pd.Series(["dga"] * len(chunk))

                out = pd.DataFrame({"domain": domains, "threat": labels})
                out = out.dropna(subset=["domain", "threat"]).copy()
                out["domain"] = out["domain"].astype(str).str.strip().str.lower()
                out["threat"] = out["threat"].astype(str).str.strip().str.lower()
                out = out[(out["threat"] == "dga") | (out["threat"] == "benign")]
                # Validate domain pattern
                mask = out["domain"].apply(is_valid_domain)
                out = out[mask]
                for row in out.itertuples(index=False):
                    yield row.domain, row.threat
            read_ok = True
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if not read_ok and last_err is not None:
        console.print(f"[yellow]Skipping {csv_path} due to read error: {last_err}")


def iter_txt_records(txt_path: str) -> Iterable[Tuple[str, str]]:
    lower_name = os.path.basename(txt_path).lower()
    if any(k in lower_name for k in ["benign", "legit", "clean", "alexa", "top"]):
        label = "benign"
    elif any(k in lower_name for k in ["dga", "malicious", "generated", "family"]):
        label = "dga"
    else:
        # Fallback: consider as dga list
        label = "dga"
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dom = line.strip().lower()
            if is_valid_domain(dom):
                yield dom, label


def find_data_files(root: str) -> Tuple[List[str], List[str]]:
    csv_files: List[str] = []
    txt_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            lower = name.lower()
            if lower.endswith(".csv"):
                csv_files.append(path)
            elif lower.endswith(".txt") or lower.endswith(".tsv"):
                txt_files.append(path)
    return csv_files, txt_files


def convert(input_dir: str, output_jsonl: str, limit: Optional[int] = None, shuffle: bool = False, seed: int = 42) -> None:
    csv_files, txt_files = find_data_files(input_dir)
    if not csv_files and not txt_files:
        raise SystemExit(f"No CSV/TXT files found under {input_dir}")

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    total_written = 0
    tmp_records: List[Tuple[str, str]] = []

    with Progress(transient=True) as progress:
        if csv_files:
            task = progress.add_task("Converting CSV files", total=len(csv_files))
            for csv_path in csv_files:
                for domain, label in iter_csv_records(csv_path):
                    tmp_records.append((domain, label))
                    if limit is not None and len(tmp_records) >= limit:
                        break
                progress.advance(task, 1)
                if limit is not None and len(tmp_records) >= limit:
                    break
        if (limit is None or len(tmp_records) < limit) and txt_files:
            task = progress.add_task("Converting TXT files", total=len(txt_files))
            for txt_path in txt_files:
                for domain, label in iter_txt_records(txt_path):
                    tmp_records.append((domain, label))
                    if limit is not None and len(tmp_records) >= limit:
                        break
                progress.advance(task, 1)
                if limit is not None and len(tmp_records) >= limit:
                    break

    if shuffle:
        import random
        rnd = random.Random(seed)
        rnd.shuffle(tmp_records)

    # Optional dedup preserving last occurrence
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for dom, lab in tmp_records:
        if dom in seen:
            continue
        seen.add(dom)
        deduped.append((dom, lab))
    tmp_records = deduped

    if limit is not None:
        tmp_records = tmp_records[:limit]

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for dom, lab in tmp_records:
            out.write(json.dumps({"domain": dom, "threat": lab}, ensure_ascii=False) + "\n")
            total_written += 1

    console.print(f"[green]Wrote {total_written:,} records to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Convert ExtraHop DGA dataset to JSONL {domain, threat}")
    parser.add_argument("--input_dir", required=True, help="Path to ExtraHop repo root or data dir")
    parser.add_argument("--output_jsonl", required=True, help="Destination JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Write at most N records (debug)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before writing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert(args.input_dir, args.output_jsonl, limit=args.limit, shuffle=args.shuffle, seed=args.seed)


if __name__ == "__main__":
    main()
