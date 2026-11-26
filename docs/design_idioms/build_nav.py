#!/usr/bin/env python3
"""Regenerate nav.js and the shared boilerplate of every page in this folder.

Run it after editing any page: ``python docs/design_idioms/build_nav.py``.
``--check`` writes nothing and exits 1 when any file would change.

What a page author edits is only the text between ``<!-- IDIOM BODY START -->`` and
``<!-- IDIOM BODY END -->``. Everything else in a page is written here: the head and
<title>, the wrapper section and its group class, the scripts, and the previous/next
links. Inside the body, only the number line ``<p class="num">NN</p>`` is rewritten.
In index.html the idiom lists between ``<!-- TOC <group> START/END -->`` and the
counts in the stats list are rewritten; the rest of index.html is hand-written.

The order and membership of the groups is GROUPS below: add a page there.
"""
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (key, rail label, numbered, page ids in order)
GROUPS = [
    ("execution", "Execution", True, [
        "e-file", "e-object", "e-purpose", "e-input", "e-words", "e-from", "e-hook",
        "e-return", "e-writes", "e-root",
    ]),
    ("planning", "Planning", True, [
        "p-plan", "p-run", "p-facts", "p-skip", "p-result", "p-fan", "p-call", "p-share",
    ]),
    ("context", "Contexts", True, [
        "c-folder", "c-base", "c-parent", "c-state", "c-identifiers",
    ]),
    ("reference", "Reference", False, ["r-turn", "r-operate", "r-shapes", "r-helpers", "r-steps", "removed"]),
]

START = "<!-- IDIOM BODY START -->"
END = "<!-- IDIOM BODY END -->"
BODY_RE = re.compile(re.escape(START) + r"(.*?)" + re.escape(END), re.S)
NUM_RE = re.compile(r'<p class="num">[^<]*</p>')
INLINE = {"a", "b", "code", "em", "i", "span", "strong", "sub", "sup", "kbd", "abbr"}

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Design idioms · fastWorkflow</title>
<link rel="stylesheet" href="idioms.css">
</head>
<body data-page="{id}">
<aside class="rail"></aside>

<main id="top">
  <section class="{wrapper}">
{start}{body}{end}
  </section>

  <nav class="pager" aria-label="Previous and next">
{pager}
  </nav>
</main>
<script src="nav.js"></script>
<script src="idioms.js"></script>
</body>
</html>
"""


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag not in INLINE:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag not in INLINE:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(fragment):
    parser = _Text()
    parser.feed(fragment)
    parser.close()
    return " ".join("".join(parser.parts).split())


def first(pattern, fragment):
    match = re.search(pattern, fragment, re.S)
    return plain_text(match.group(1)) if match else None


def read_body(path):
    if not path.exists():
        raise SystemExit(f"build_nav: {path.name} is listed in GROUPS but does not exist")
    found = BODY_RE.findall(path.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise SystemExit(f"build_nav: {path.name} must hold exactly one {START} ... {END} pair")
    return found[0]


def pager_link(kind, label, page):
    if page is None:
        return ""
    return (f'    <a class="{kind}" href="{page["id"]}.html"><span>{label}</span>'
            f'{html.escape(page["title"], quote=False)}</a>')


def build():
    pages = []
    for key, _label, numbered, ids in GROUPS:
        for n, page_id in enumerate(ids, 1):
            body = read_body(HERE / f"{page_id}.html")
            number = f"{n:02d}" if numbered else ""
            if numbered:
                body = NUM_RE.sub(f'<p class="num">{number}</p>', body, count=1)
            title = first(r"<h3[^>]*>(.*?)</h3>", body) or first(r'<p class="stamp">(.*?)</p>', body)
            if not title:
                raise SystemExit(f"build_nav: {page_id}.html has no <h3> or stamp to title it")
            pages.append({"id": page_id, "group": key, "number": number, "title": title,
                          "text": plain_text(body), "body": body})

    listed = {p["id"] for p in pages}
    strays = sorted(p.stem for p in HERE.glob("*.html") if p.stem != "index" and p.stem not in listed)
    if strays:
        raise SystemExit(f"build_nav: pages not listed in GROUPS: {', '.join(strays)}")

    out = {}
    for i, page in enumerate(pages):
        prev_page = pages[i - 1] if i > 0 else None
        next_page = pages[i + 1] if i + 1 < len(pages) else None
        wrapper = ("reference-block reference" if page["group"] == "reference"
                   else f"group {page['group']}")
        pager = "\n".join(x for x in (pager_link("prev", "Previous", prev_page),
                                     pager_link("next", "Next", next_page)) if x)
        out[HERE / f"{page['id']}.html"] = PAGE.format(
            title=html.escape(page["title"], quote=False), id=page["id"], wrapper=wrapper,
            start=START, body=page["body"], end=END, pager=pager)

    nav_lines = ["// Generated by build_nav.py from the pages in this folder. Do not edit.",
                 "window.IDIOMS_NAV = {", '  "groups": [']
    nav_lines.append(",\n".join(
        "    " + json.dumps({"key": k, "label": label, "counted": numbered})
        for k, label, numbered, _ in GROUPS))
    nav_lines += ["  ],", '  "pages": [']
    nav_lines.append(",\n".join(
        "    " + json.dumps({k: p[k] for k in ("id", "group", "number", "title", "text")})
        for p in pages))
    nav_lines += ["  ]", "};", ""]
    out[HERE / "nav.js"] = "\n".join(nav_lines)

    index_path = HERE / "index.html"
    index = index_path.read_text(encoding="utf-8")
    for key, _label, numbered, ids in GROUPS:
        rows = []
        for p in (p for p in pages if p["group"] == key):
            num = f'<span class="num">{p["number"]}</span>' if p["number"] else ""
            rows.append(f'      <li data-id="{p["id"]}"><a href="{p["id"]}.html">{num}'
                        f'{html.escape(p["title"], quote=False)}</a></li>')
        block = (f"<!-- TOC {key} START -->\n    <ol class=\"toc\">\n" + "\n".join(rows)
                 + f"\n    </ol>\n    <!-- TOC {key} END -->")
        pattern = re.compile(rf"<!-- TOC {key} START -->.*?<!-- TOC {key} END -->", re.S)
        if not pattern.search(index):
            raise SystemExit(f"build_nav: index.html has no TOC markers for {key}")
        index = pattern.sub(lambda _m: block, index)
        if numbered:
            index = re.sub(rf'(<li class="{key}"><b>)\d+(</b>)', rf"\g<1>{len(ids)}\g<2>", index)
    out[index_path] = index
    return out


def main(argv):
    check = "--check" in argv
    outputs = build()
    changed = [path for path, text in outputs.items()
               if not path.exists() or path.read_text(encoding="utf-8") != text]
    if check:
        for path in changed:
            print(f"would change: {path.name}")
        return 1 if changed else 0
    for path in changed:
        path.write_text(outputs[path], encoding="utf-8")
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
