#!/usr/bin/env python3
"""
Download Brazilian state flags (SVG) from the Wikimedia Commons gallery page.

Usage examples:
  python get_br_flags.py
  python get_br_flags.py --page "https://commons.wikimedia.org/wiki/Flags_of_states_of_Brazil" --outdir ./flags-br --dry-run

The script finds <img> tags that use Wikimedia's `/thumb/` URLs and
reconstructs the original `.svg` URL by removing `/thumb` and trimming
after the first `.svg` occurrence. e.g.:

  https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Bandeira_do_Amap%C3%A1.svg/250px-Bandeira_do_Amap%C3%A1.svg.png

becomes

  https://upload.wikimedia.org/wikipedia/commons/0/0c/Bandeira_do_Amap%C3%A1.svg

The script saves files into the output directory and avoids duplicate downloads.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from urllib.parse import unquote, urljoin
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess

try:
    import requests
    from bs4 import BeautifulSoup
except Exception as e:
    print("Missing dependency: please install requirements: pip install requests beautifulsoup4", file=sys.stderr)
    raise


WIKI_DEFAULT = "https://commons.wikimedia.org/wiki/Flags_of_states_of_Brazil"

# Use a common browser User-Agent so Wikimedia doesn't reject the request
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    )
}


def fetch_with_retries(url: str, session: requests.sessions.Session | None = None, *,
                       max_attempts: int = 3, timeout: int = 30, headers: dict | None = None) -> requests.Response:
    headers = headers or DEFAULT_HEADERS
    sess = session or requests
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = sess.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            # exponential-ish backoff
            time.sleep(0.5 * attempt)
    raise last_exc


def find_svg_urls_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml") if "lxml" in sys.modules else BeautifulSoup(html, "html.parser")
    urls = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        # Skip tiny icons / icons.wikimedia etc
        if "upload.wikimedia.org" not in src:
            continue
        # If it's a thumb URL, reconstruct the original svg (we only want .svg files)
        if "/thumb/" in src:
            prefix, rest = src.split("/thumb/", 1)
            if ".svg" in rest:
                img_path = rest.split(".svg", 1)[0] + ".svg"
                img_url = prefix + "/" + img_path
                urls.append(img_url)
            else:
                # no svg in this thumb URL — skip
                continue
        else:
            # direct link — keep only .svg
            if ".svg" in src:
                parts = re.split(r"(\.svg)", src, maxsplit=1)
                if len(parts) >= 3:
                    urls.append(parts[0] + parts[1])

    # Normalize / de-duplicate and return
    normalized = []
    seen = set()
    for u in urls:
        # ensure full schema
        if u.startswith("//"):
            u = "https:" + u
        if u.startswith("/"):
            u = urljoin("https://commons.wikimedia.org", u)
        if u not in seen:
            seen.add(u)
            normalized.append(u)
    return normalized


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# 27 federative units (26 states + Distrito Federal)
_BRAZIL_UNITS = [
    "Acre",
    "Alagoas",
    "Amapa",
    "Amazonas",
    "Bahia",
    "Ceara",
    "Distrito Federal",
    "Espirito Santo",
    "Goias",
    "Maranhao",
    "Mato Grosso",
    "Mato Grosso do Sul",
    "Minas Gerais",
    "Para",
    "Paraiba",
    "Parana",
    "Pernambuco",
    "Piaui",
    "Rio de Janeiro",
    "Rio Grande do Norte",
    "Rio Grande do Sul",
    "Rondonia",
    "Roraima",
    "Santa Catarina",
    "Sao Paulo",
    "Sergipe",
    "Tocantins",
]

_NORM_UNITS = [_normalize_text(x) for x in _BRAZIL_UNITS]

# Two-letter state codes (ISO 3166-2:BR style) in lowercase, mapped by normalized unit name
_UNIT_CODE_MAP = {
    _normalize_text("Acre"): "ac",
    _normalize_text("Alagoas"): "al",
    _normalize_text("Amapa"): "ap",
    _normalize_text("Amazonas"): "am",
    _normalize_text("Bahia"): "ba",
    _normalize_text("Ceara"): "ce",
    _normalize_text("Distrito Federal"): "df",
    _normalize_text("Espirito Santo"): "es",
    _normalize_text("Goias"): "go",
    _normalize_text("Maranhao"): "ma",
    _normalize_text("Mato Grosso"): "mt",
    _normalize_text("Mato Grosso do Sul"): "ms",
    _normalize_text("Minas Gerais"): "mg",
    _normalize_text("Para"): "pa",
    _normalize_text("Paraiba"): "pb",
    _normalize_text("Parana"): "pr",
    _normalize_text("Pernambuco"): "pe",
    _normalize_text("Piaui"): "pi",
    _normalize_text("Rio de Janeiro"): "rj",
    _normalize_text("Rio Grande do Norte"): "rn",
    _normalize_text("Rio Grande do Sul"): "rs",
    _normalize_text("Rondonia"): "ro",
    _normalize_text("Roraima"): "rr",
    _normalize_text("Santa Catarina"): "sc",
    _normalize_text("Sao Paulo"): "sp",
    _normalize_text("Sergipe"): "se",
    _normalize_text("Tocantins"): "to",
}


def is_federative_unit_image(url: str) -> bool:
    # accept .svg or .png
    if not (url.lower().endswith(".svg") or url.lower().endswith(".png")):
        return False
    name = filename_from_url(url)
    name = unquote(name)
    name_norm = _normalize_text(name)
    # require that the filename mentions 'bandeira' or 'flag' as a hint
    name_lower = name.lower()
    if "bandeira" not in name_norm and "flag" not in name_norm and "bandeira" not in name_lower:
        # still allow if a state name is present
        pass
    # Exclude historical variants: filenames containing a 4-digit year or words indicating historical/old flags
    # e.g. 'Bandeira_de_Alagoas_1889.svg', 'Bandeira_antiga_do_X.svg', 'old', 'former', 'histor'
    # Use lookaround to match years even when adjacent to underscores or other word characters
    if re.search(r"(?<!\d)\d{4}(?!\d)", name):
        return False
    if any(w in name_lower for w in ("antiga", "antigo", "histor", "former", "old", "pre-", "provinc", "imperial")):
        return False
    for unit in _NORM_UNITS:
        if unit in name_norm:
            return True
    return False


def filename_from_url(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return unquote(name)


def download_url(url: str, outpath: str) -> bool:
    try:
        r = fetch_with_retries(url, max_attempts=4, timeout=30, headers=DEFAULT_HEADERS)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

    try:
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Failed to write file {outpath}: {e}")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Download Brazilian state flag SVGs from Wikimedia Commons gallery")
    p.add_argument("--page", default=WIKI_DEFAULT, help="Wikimedia gallery page URL")
    p.add_argument("--outdir", default="flags-br", help="Output directory")
    p.add_argument("--limit", type=int, default=0, help="Limit number of flags to download (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Only print discovered image URLs")
    p.add_argument("--no-download", action="store_true", help="Don't download; same as --dry-run")
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Include all discovered SVGs. By default the script only keeps "
            "Brazil's 27 federative units (states + Distrito Federal)."
        ),
    )
    p.add_argument("--no-convert", action="store_true", help="Do not convert downloaded SVGs to PNG (conversion is performed by default)")
    p.add_argument("--png-dest", help="Destination directory for generated PNGs (optional)")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing PNG files when converting (overwrite by default)")
    p.add_argument("--scale", type=float, default=1.0, help="Scale multiplier for rasterization (default: 1.0)")
    p.add_argument("--width", type=int, default=None, help="Output width in pixels (overrides scale if provided)")
    p.add_argument("--height", type=int, default=None, help="Output height in pixels (overrides scale if provided)")
    p.add_argument("--max-dimension", type=int, default=None, help="Maximum pixel dimension for the largest side; preserves aspect ratio")
    p.add_argument("--workers", "-w", type=int, default=4, help="Number of worker threads for conversion (default: 4)")
    p.add_argument("--quiet", "-q", action="store_true", help="Quiet mode; minimal output for conversion")
    p.add_argument("--inkscape-path", help="Path to inkscape executable or folder containing it")
    args = p.parse_args()

    print(f"Fetching page: {args.page}")
    try:
        resp = fetch_with_retries(args.page, max_attempts=4, timeout=30, headers=DEFAULT_HEADERS)
    except Exception as e:
        print(f"Failed to fetch page: {e}", file=sys.stderr)
        return 2

    svg_urls = find_svg_urls_from_html(resp.text)

    # keep only .svg
    svg_urls = [u for u in svg_urls if u.lower().endswith(".svg")]

    def is_historical_filename(name: str) -> bool:
        nl = name.lower()
        if re.search(r"\b\d{4}\b", nl):
            return True
        if any(w in nl for w in ("antiga", "antigo", "histor", "former", "old", "pre-", "provincia", "província", "imperial", "provinc")):
            return True
        return False

    unit_url_map: dict[str, str] = {}
    if not args.all:
        # For each federative unit, choose the best matching SVG (exclude historicals)
        for unit_norm in _NORM_UNITS:
            # candidates whose normalized filename contains the unit token
            candidates = []
            for u in svg_urls:
                fname = filename_from_url(u)
                fname_norm = _normalize_text(unquote(fname))
                if unit_norm in fname_norm:
                    candidates.append(u)

            if not candidates:
                continue

            nonhist = [u for u in candidates if not is_historical_filename(filename_from_url(u))]
            pool = nonhist if nonhist else candidates

            best = None
            best_score = -10_000
            for u in pool:
                fname = filename_from_url(u)
                nl = fname.lower()
                score = 0
                if "bandeira" in nl or "flag" in nl:
                    score += 50
                if "estadual" in nl:
                    score -= 5
                if any(x in nl for x in ("provincia", "província", "provinc")):
                    score -= 50
                if re.search(r"\b\d{4}\b", nl):
                    score -= 100
                score += max(0, 30 - len(nl))
                if score > best_score:
                    best_score = score
                    best = u

            if best:
                code = _UNIT_CODE_MAP.get(unit_norm)
                if code:
                    unit_url_map[code] = best

        # svg_urls will be the list of URLs selected (for backwards compatibility below)
        svg_urls = list(unit_url_map.values())

    if args.limit > 0:
        svg_urls = svg_urls[: args.limit]

    if not svg_urls:
        print("No SVG URLs found.")
        return 0

    print(f"Found {len(svg_urls)} svg candidates (unique).")

    for u in svg_urls:
        print(u)
    if args.dry_run or args.no_download:
        print("Dry run; not downloading files.")
        return 0

    os.makedirs(args.outdir, exist_ok=True)

    downloaded = 0
    # build reverse map from url->code for saving
    url_to_code = {v: k for k, v in unit_url_map.items()} if 'unit_url_map' in locals() else {}
    for u in svg_urls:
        fname = filename_from_url(u)
        code = url_to_code.get(u)
        if code:
            outname = f"{code.upper()}.svg"
        else:
            outname = fname
        outpath = os.path.join(args.outdir, outname)
        if os.path.exists(outpath):
            print(f"Skipping, exists: {outname}")
            continue
        print(f"Downloading {outname} ... ")
        ok = download_url(u, outpath)
        if ok:
            downloaded += 1
        # be nice to servers
    print(f"Downloaded {downloaded} files to {args.outdir}")
    # By default convert downloaded SVGs to PNG unless user disabled it
    # Decide overwrite behavior: default is True unless user passes --no-overwrite
    overwrite = not getattr(args, "no_overwrite", False)

    if not args.no_convert:
        # find inkscape
        inkscape_path = None
        if args.inkscape_path:
            inkscape_path = args.inkscape_path
        if not inkscape_path:
            inkscape_path = os.environ.get("INKSCAPE")
        if not inkscape_path:
            inkscape_path = shutil.which("inkscape") or shutil.which("inkscape.exe")
        if inkscape_path:
            p = Path(inkscape_path)
            if p.is_dir():
                candidate = p / "inkscape.exe"
                if candidate.exists():
                    inkscape_path = str(candidate)
        if not inkscape_path:
            # try common Windows location
            hardcoded = r"C:\Program Files\Inkscape\bin\inkscape.exe"
            if Path(hardcoded).exists():
                inkscape_path = hardcoded
        if not inkscape_path:
            print("Inkscape CLI not found. Provide --inkscape-path or add Inkscape to PATH.", file=sys.stderr)
            return 2

        source = Path(args.outdir).resolve()
        if not source.exists() or not source.is_dir():
            print(f"Source directory not found or not a directory: {source}", file=sys.stderr)
            return 2

        if args.png_dest:
            dest = Path(args.png_dest).resolve()
        else:
            dest = source.with_name(source.name + "_png")

        # conversion helper functions
        def find_svgs(source_dir: Path):
            # Prefer code-named svgs in the root of source_dir if we have unit_url_map
            if url_to_code:
                for code in url_to_code.values():
                    p = source_dir / f"{code.upper()}.svg"
                    if p.exists() and p.is_file():
                        yield p
                # also yield any other svg files found recursively
            for p in source_dir.rglob("*.svg"):
                if p.is_file():
                    yield p

        def should_convert(src: Path, dst: Path, overwrite: bool) -> bool:
            if overwrite:
                return True
            if not dst.exists():
                return True
            try:
                return src.stat().st_mtime > dst.stat().st_mtime
            except OSError:
                return True

        def _detect_svg_ratio(src: Path) -> tuple[float, float] | None:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(src)
                root = tree.getroot()
                vb = root.get("viewBox")
                if vb:
                    parts = vb.replace(",", " ").split()
                    if len(parts) >= 4:
                        vb_w = float(parts[2])
                        vb_h = float(parts[3])
                        return vb_w, vb_h
                w = root.get("width")
                h = root.get("height")
                if w and h:
                    def parse_val(v: str) -> float:
                        import re as _re
                        m = _re.match(r"([0-9.]+)", v)
                        return float(m.group(1)) if m else None
                    wv = parse_val(w)
                    hv = parse_val(h)
                    if wv and hv:
                        return float(wv), float(hv)
            except Exception:
                return None
            return None

        def convert_one(src: Path, dst: Path, width: int | None, height: int | None, scale: float, overwrite: bool, inkscape_path: str, max_dimension: int | None) -> tuple[Path, bool, str]:
            if not should_convert(src, dst, overwrite):
                return src, True, "skipped (up-to-date)"
            dst.parent.mkdir(parents=True, exist_ok=True)
            use_width = width
            use_height = height
            if max_dimension is not None and width is None and height is None:
                dims = _detect_svg_ratio(src)
                if dims:
                    wv, hv = dims
                    if wv > hv:
                        use_width = max_dimension
                    else:
                        use_height = max_dimension
                else:
                    use_width = max_dimension

            cmd = [inkscape_path]
            cmd += [str(src)]
            cmd += ["--export-filename", str(dst)]
            if use_width is not None:
                cmd += ["--export-width", str(int(use_width))]
            if use_height is not None:
                cmd += ["--export-height", str(int(use_height))]

            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if proc.returncode == 0:
                    os.utime(dst, (src.stat().st_atime, src.stat().st_mtime))
                    return src, True, f"converted -> {dst}"
                else:
                    return src, False, f"inkscape failed (code {proc.returncode}): {proc.stderr.decode(errors='replace')}"
            except FileNotFoundError as e:
                return src, False, f"inkscape not found: {e}"
            except Exception as e:
                return src, False, f"error: {e}"

        start = time.time()
        svgs = list(find_svgs(source))
        if not args.quiet:
            print(f"Found {len(svgs)} .svg files under {source}")
            print(f"PNG Destination directory: {dest}")

        converted = 0
        skipped = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for svg in svgs:
                try:
                    rel = svg.relative_to(source)
                except Exception:
                    rel = svg.name
                # if svg is root-level code-named, place png in dest root with same name
                if svg.parent == source:
                    out_png = dest.joinpath(svg.name).with_suffix('.png')
                else:
                    out_png = dest.joinpath(rel).with_suffix('.png')
                futures[ex.submit(convert_one, svg, out_png, args.width, args.height, args.scale, overwrite, inkscape_path, args.max_dimension)] = svg

            for fut in as_completed(futures):
                src, ok, msg = fut.result()
                if args.quiet:
                    continue
                if ok:
                    if msg.startswith('skipped'):
                        skipped += 1
                    else:
                        converted += 1
                    print(f"{src} -> {msg}")
                else:
                    failed += 1
                    print(f"{src} -> {msg}")

        elapsed = time.time() - start
        if not args.quiet:
            print(f"Done: converted={converted}, skipped={skipped}, failed={failed} in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
