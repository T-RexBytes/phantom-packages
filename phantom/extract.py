"""Extract suggested PyPI distribution names from a model's reply (BUILD_SPEC §5.3, §6).

Priority: install lines (pip / python -m pip / py -m pip / uv pip / uv add),
then requirements blocks, then imports mapped through ``IMPORT_TO_DIST``.
Names are PEP 503-normalized and deduplicated per reply (first source wins).
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field

# Import name -> distribution name, only where the two differ after PEP 503.
IMPORT_TO_DIST: dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "Cryptodome": "pycryptodomex",
    "serial": "pyserial",
    "usb": "pyusb",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "magic": "python-magic",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "OpenSSL": "pyopenssl",
    "attr": "attrs",
    "google.protobuf": "protobuf",
    "googleapiclient": "google-api-python-client",
    "google.auth": "google-auth",
    "google.oauth2": "google-auth",
    "skimage": "scikit-image",
    "Bio": "biopython",
    "fitz": "pymupdf",
    "win32api": "pywin32",
    "win32com": "pywin32",
    "win32con": "pywin32",
    "win32evtlog": "pywin32",
    "pythoncom": "pywin32",
    "zmq": "pyzmq",
    "gi": "pygobject",
    "MySQLdb": "mysqlclient",
    "mysql": "mysql-connector-python",
    "osgeo": "gdal",
    "shapefile": "pyshp",
    "mpl_toolkits": "matplotlib",
    "pkg_resources": "setuptools",
    "telegram": "python-telegram-bot",
    "discord": "discord-py",
    "dns": "dnspython",
    "git": "gitpython",
    "github": "pygithub",
    "jose": "python-jose",
    "multipart": "python-multipart",
    "socketio": "python-socketio",
    "can": "python-can",
    "nmap": "python-nmap",
    "whois": "python-whois",
    "ldap": "python-ldap",
    "ffmpeg": "ffmpeg-python",
    "vlc": "python-vlc",
    "rtmidi": "python-rtmidi",
    "speech_recognition": "speechrecognition",
    "RPi": "rpi-gpio",
    "board": "adafruit-blinka",
    "busio": "adafruit-blinka",
    "paho": "paho-mqtt",
    "bluetooth": "pybluez",
    "Evtx": "python-evtx",
    "yara": "yara-python",
    "odf": "odfpy",
    "pdfminer": "pdfminer-six",
    "camelot": "camelot-py",
    "tabula": "tabula-py",
    "vcf": "pyvcf",
    "stdnum": "python-stdnum",
    "omg": "omgifol",
    "Xlib": "python-xlib",
    "wx": "wxpython",
    "cairo": "pycairo",
    "OpenGL": "pyopengl",
    "kafka": "kafka-python",
    "memcache": "python-memcached",
    "nacl": "pynacl",
    "slugify": "python-slugify",
    "decouple": "python-decouple",
    "bson": "pymongo",
    "ruamel": "ruamel-yaml",
}

# Import names that look like the user's own modules, not distributions.
LOCAL_MODULES = frozenset(
    {"utils", "util", "config", "settings", "helpers", "helper", "models", "main", "app",
     "src", "lib", "common", "constants", "tests", "test", "module", "mymodule", "package"}
)

# A word that ends a prose-like argument list ("pip install x then run ...").
STOPWORDS = frozenset(
    {"and", "then", "to", "or", "for", "first", "if", "before", "after", "with", "in", "on",
     "the", "a", "an", "it", "this", "that", "run", "using", "via", "from", "is", "are",
     "package", "packages", "library", "libraries", "module", "command", "instead"}
)

# Flags whose next token is a value, not a package.
FLAGS_WITH_VALUE = frozenset(
    {"-i", "--index-url", "--extra-index-url", "-r", "--requirement", "-e", "--editable",
     "-c", "--constraint", "-f", "--find-links", "-t", "--target", "--prefix", "--root",
     "--trusted-host", "--platform", "--python-version", "--implementation", "--abi",
     "--only-binary", "--no-binary", "--progress-bar", "--log", "--cache-dir", "--src",
     "--upgrade-strategy", "--python", "-p", "--group", "--optional", "--extra", "--index",
     "--default-index"}
)

STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}
REQUIREMENTS_LANGS = frozenset({"requirements.txt", "requirements", "pip-requirements", "reqs"})
PLAIN_LANGS = frozenset({"", "text", "txt", "plaintext", "plain"})

_COMMAND = re.compile(
    r"(?:^|(?<=[\s!%$>]))"
    r"(?:(?P<uv>uv\s+pip\s+install|uv\s+add)"
    r"|(?P<pip>(?:pip3?|python[\d.]*\s+-m\s+pip|py\s+-m\s+pip)\s+install))"
    r"(?=\s)"
)
_PROSE_COMMAND = re.compile(r"^\s*(?:\$\s*)?(?:pip3?|python[\d.]*\s+-m|py\s+-m|uv)\s")
_LEADING_MARKER = re.compile(r"^\s*(?:#+|//|>+|\$|PS>|REM\s)\s*")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^\s*(```|~~~)\s*([^\s`]*)")
_IMPORT = re.compile(r"^\s*import\s+(.+)$")
_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import\b")
_NAME_END = re.compile(r"[\[<>=!~;@\s(,]")
_VALID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQ_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?\s*"
    r"(?:(?:==|>=|<=|~=|!=|<|>)\s*[\w.*+!-]+\s*,?\s*)*(?:;.*)?$"
)


def normalize(name: str) -> str:
    """PEP 503 normalization: runs of ``-_.`` become ``-``, lowercased."""
    return re.sub(r"[-_.]+", "-", name).lower()


_MODULE_CONFUSION = frozenset(normalize(k) for k in IMPORT_TO_DIST)
_STDLIB_NORMALIZED = frozenset(normalize(m) for m in STDLIB)


def subtype(name: str) -> str | None:
    """Return ``module_name_confusion``, ``stdlib_confusion`` or None for a pip name."""
    n = normalize(name)
    if n in _MODULE_CONFUSION:
        return "module_name_confusion"
    if n in _STDLIB_NORMALIZED:
        return "stdlib_confusion"
    return None


@dataclass(frozen=True)
class Mention:
    """One extracted name. ``source`` is pip_line, uv_line, requirements_block,
    import_mapped or import_unmapped (the last is counted but never labelled)."""

    raw_token: str
    source: str
    name: str
    position: int


@dataclass
class Extraction:
    """All mentions of one reply plus raw tokens dropped by the validity filter."""

    mentions: list[Mention] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


@dataclass
class _Block:
    lang: str
    lines: list[str]


def _split_markdown(text: str) -> tuple[list[str], list[_Block]]:
    """Split into prose lines and fenced code blocks (an unclosed fence runs to the end)."""
    prose: list[str] = []
    blocks: list[_Block] = []
    current: _Block | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if current is None and m:
            current = _Block(m.group(2).lower(), [])
        elif current is not None and line.strip() in ("```", "~~~"):
            blocks.append(current)
            current = None
        elif current is not None:
            current.lines.append(line)
        else:
            prose.append(line)
    if current is not None:
        blocks.append(current)
    return prose, blocks


def _join_continuations(lines: list[str]) -> list[str]:
    """Join shell line continuations ending in ``\\`` or `` ^``."""
    out: list[str] = []
    buf = ""
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("\\") or stripped.endswith(" ^"):
            buf += stripped[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _split_unquoted(s: str) -> list[str]:
    """Split on unquoted ``&& || ; | ( )`` and backticks; drop an unquoted `` #`` comment."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == quote:
                quote = None
            buf.append(ch)
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "#" and (i == 0 or s[i - 1].isspace()):
            break
        elif ch in ";|&()`":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def _tokens(args: str) -> list[str]:
    try:
        return shlex.split(args, posix=True)
    except ValueError:
        return [t.strip("'\"") for t in args.split()]


def _to_name(token: str) -> str:
    """Strip extras, version specifiers, markers and trailing punctuation from a token."""
    token = token.strip().strip("'\"")
    m = _NAME_END.search(token)
    base = token[: m.start()] if m else token
    return re.sub(r"[^A-Za-z0-9]+$", "", base)


def _is_location(token: str) -> bool:
    return (
        "://" in token or "/" in token or "\\" in token or token.startswith("git+")
        or token in {".", "..", "@"} or token.endswith((".whl", ".tar.gz", ".zip"))
    )


def _command_names(line: str) -> list[tuple[str, str, list[tuple[str, str]], list[str]]]:
    """Parse install commands in one logical line -> [(source, cmd, [(raw, name)], dropped)]."""
    found = []
    line = _LEADING_MARKER.sub("", line)
    for segment in _split_unquoted(line):
        matches = list(_COMMAND.finditer(segment))
        for idx, m in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(segment)
            source = "uv_line" if m.group("uv") else "pip_line"
            kept: list[tuple[str, str]] = []
            dropped: list[str] = []
            skip_value = False
            for tok in _tokens(segment[m.end():end]):
                if skip_value:
                    skip_value = False
                    continue
                if tok.startswith("-"):
                    skip_value = tok in FLAGS_WITH_VALUE
                    continue
                if tok.lower() in STOPWORDS:
                    break
                if _is_location(tok):
                    continue
                name = _to_name(tok)
                if not name and not tok.strip("'\""):
                    continue
                if not _VALID.match(name) or len(name) > 80 or name != tok.strip("'\"")[: len(name)]:
                    dropped.append(tok)
                    continue
                kept.append((tok, name))
            found.append((source, m.group(0), kept, dropped))
    return found


def _import_modules(line: str) -> list[tuple[str, list[str]]]:
    """Return [(raw_token, dotted candidates longest-first)] for one import line."""
    out = []
    if m := _FROM.match(line):
        mod = m.group(1)
        if not mod.startswith("."):
            out.append((f"from {mod} import", _prefixes(mod)))
    elif m := _IMPORT.match(line):
        for part in m.group(1).split(","):
            mod = part.strip().split(" as ")[0].strip()
            if re.fullmatch(r"[A-Za-z_][\w.]*", mod):
                out.append((f"import {mod}", _prefixes(mod)))
    return out


def _prefixes(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def extract(text: str) -> Extraction:
    """Extract every suggested distribution name from one model reply."""
    result = Extraction()
    seen: set[str] = set()

    def add(raw: str, source: str, name: str, position: int) -> None:
        n = normalize(name)
        if n not in seen:
            seen.add(n)
            result.mentions.append(Mention(raw, source, n, position))

    prose, blocks = _split_markdown(text)
    req_blocks = [b for b in blocks if _is_requirements(b)]
    code_blocks = [b for b in blocks if b not in req_blocks]

    command_lines: list[str] = []
    for b in code_blocks:
        command_lines += _join_continuations(b.lines)
    for line in prose:
        command_lines += _INLINE_CODE.findall(line)
        if _PROSE_COMMAND.match(line):
            command_lines.append(_INLINE_CODE.sub(" ", line))

    for line in command_lines:
        for source, _cmd, kept, dropped in _command_names(line):
            result.dropped += dropped
            for pos, (raw, name) in enumerate(kept):
                add(raw, source, name, pos)

    for b in req_blocks:
        pos = 0
        for line in b.lines:
            s = line.split("#", 1)[0].strip()
            if not s or s.startswith("-"):
                continue
            name = _to_name(s)
            if _VALID.match(name) and len(name) <= 80:
                add(s, "requirements_block", name, pos)
                pos += 1
            else:
                result.dropped.append(s)

    for b in code_blocks:
        for line in b.lines:
            for pos, (raw, candidates) in enumerate(_import_modules(line)):
                top = candidates[-1]
                if top in STDLIB or top.lower() in LOCAL_MODULES:
                    continue
                mapped = next((IMPORT_TO_DIST[c] for c in candidates if c in IMPORT_TO_DIST), None)
                if mapped:
                    add(raw, "import_mapped", mapped, pos)
                elif _VALID.match(top):
                    add(raw, "import_unmapped", top, pos)
    return result


def _is_requirements(block: _Block) -> bool:
    if block.lang in REQUIREMENTS_LANGS or block.lang.endswith("requirements.txt"):
        return True
    if block.lang not in PLAIN_LANGS:
        return False
    lines = [ln.split("#", 1)[0].strip() for ln in block.lines]
    lines = [ln for ln in lines if ln]
    return bool(lines) and all(_REQ_LINE.match(ln) for ln in lines)
