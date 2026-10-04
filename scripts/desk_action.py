#!/usr/bin/env python3
"""Posting Desk mechanical action: post or unpost a topic's notes.

Usage: desk_action.py post-notes|unpost-notes <topic-xml-id>

Toggles the topic's <xi:include> between live and the UNPOSTED comment
wrapper defined in CLAUDE.md. Unposting also wraps any exercise-set
include whose cross-references would dangle (marked UNPOSTED-WITH so the
matching post restores it); if a dangling reference cannot be fixed by
wrapping an include, the script restores every file it touched and exits
nonzero so nothing half-done gets committed.

Exit 0: repo now in the requested state (possibly with no change).
Exit 1: bad invocation. Exit 2: cannot be done mechanically.
"""
import os
import re
import sys

SRC = "source"

# topic xml:id -> (file holding its include, include href)
TOPICS = {
    "chapter-hyperbolic-functions": ("source/main.ptx", "ch-7-3.ptx"),
    "chapter-infinite-series": ("source/ch-series-sequences.ptx", "ch-10-2.ptx"),
    "chapter-taylor-maclaurin": ("source/ch-series-sequences.ptx", "ch-10-8.ptx"),
    "section-taylor": ("source/ch-series-sequences.ptx", "Taylor.ptx"),
    "chapter-binomial-series": ("source/ch-series-sequences.ptx", "ch-10-10.ptx"),
    "chapter-three-dimensional-coordinates": ("source/ch-geometry.ptx", "ch-12-1.ptx"),
    "chapter-vectors": ("source/ch-geometry.ptx", "ch-12-2.ptx"),
    "section-dot-product": ("source/ch-geometry.ptx", "Dotproduct.ptx"),
    "section-cross-product": ("source/ch-geometry.ptx", "the-cross-product.ptx"),
    "section-parametrization": ("source/ch-geometry.ptx", "Parametrization.ptx"),
    "chapter-lines-planes": ("source/ch-geometry.ptx", "ch-12-5.ptx"),
    "section-conics": ("source/ch-geometry.ptx", "Conics.ptx"),
    "section-quadric-surfaces": ("source/ch-geometry.ptx", "QuadricSurface.ptx"),
    "chapter-functions-several-variables": ("source/ch-functions-multiple-variables.ptx", "ch-14-1.ptx"),
    "chapter-limits-continuity": ("source/ch-functions-multiple-variables.ptx", "ch-14-2.ptx"),
    "chapter-partial-derivatives": ("source/ch-functions-multiple-variables.ptx", "ch-14-3.ptx"),
    "chapter-chain-rule": ("source/ch-functions-multiple-variables.ptx", "ch-14-4.ptx"),
    "section-directional-derivatives": ("source/ch-functions-multiple-variables.ptx", "directional-derivatives.ptx"),
    "section-tangent-planes-differentials": ("source/ch-functions-multiple-variables.ptx", "TangentPlanesDifferentials.ptx"),
    "section-extreme-values": ("source/ch-functions-multiple-variables.ptx", "extreme-values-saddle-points.ptx"),
    "section-lagrange-multipliers": ("source/ch-functions-multiple-variables.ptx", "LagrangeMultipliers.ptx"),
}


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def inc_re(href):
    return re.compile(r'<xi:include\s+href="\.?/?%s"\s*/>' % re.escape(href))


def wrapped_re(href):
    return re.compile(
        r'<!--\s*UNPOSTED(?:-WITH topic="([^"]*)")?\s+'
        r'(<xi:include\s+href="\.?/?%s"\s*/>)\s+UNPOSTED\s*-->' % re.escape(href)
    )


PUBLICATION = "publication/publication.ptx"


def published_components():
    """The @component values the web publication keeps (<version include>),
    or None when the publication file elects no version scheme."""
    if not os.path.exists(PUBLICATION):
        return None
    m = re.search(r'<version\s+include="([^"]*)"', read(PUBLICATION))
    return set(m.group(1).split()) if m else None


def drop_hidden_components(text):
    """Remove every element whose @component the publication does not keep,
    as PreTeXt's assembly does, so that neither its cross-references nor its
    xml:ids take part in the dangling-reference check.  Elements are matched
    by tag name, with nesting."""
    keep = published_components()
    if keep is None:
        return text
    out, pos = [], 0
    open_tag = re.compile(r'<([A-Za-z][\w:-]*)\b[^>]*\bcomponent="([^"]*)"[^>]*>')
    for m in open_tag.finditer(text):
        if m.start() < pos or m.group(2).strip() in keep:
            continue
        end = m.end()
        if not m.group(0).endswith("/>"):
            tag = re.compile(r"<(/?)%s\b[^>]*?(/?)>" % re.escape(m.group(1)))
            depth = 1
            while depth:
                n = tag.search(text, end)
                if not n:
                    end = len(text)
                    break
                end = n.end()
                if n.group(2) == "/":
                    continue
                depth += -1 if n.group(1) else 1
        out.append(text[pos:m.start()])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def expand(path, depth=0):
    """Inline xi:includes recursively, dropping XML comments (so wrapped
    includes vanish), for cross-reference checking."""
    if depth > 20 or not os.path.exists(path):
        return ""
    text = re.sub(r"<!--.*?-->", "", read(path), flags=re.S)
    text = drop_hidden_components(text)

    def repl(m):
        return expand(os.path.join(SRC, m.group(1)), depth + 1)

    return re.sub(r'<xi:include\s+href="\.?/?([^"]+)"\s*/>', repl, text)


def dangling_refs():
    full = expand(os.path.join(SRC, "main.ptx"))
    ids = set(re.findall(r'xml:id="([^"]+)"', full))
    refs = set()
    for group in re.findall(r'<xref\b[^>]*\bref="([^"]+)"', full):
        for r in re.split(r"[,\s]+", group):
            if r:
                refs.add(r)
    return {r for r in refs if r not in ids}


def unpost_include(path, href, topic=None):
    """Wrap one include. Returns 'done' or 'already'."""
    text = read(path)
    if wrapped_re(href).search(text):
        return "already"
    m = inc_re(href).search(text)
    if m is None:
        sys.exit("include %s not found in %s" % (href, path))
    tag = "UNPOSTED" if topic is None else 'UNPOSTED-WITH topic="%s"' % topic
    write(path, text[: m.start()] + "<!-- %s %s UNPOSTED -->" % (tag, m.group(0)) + text[m.end():])
    return "done"


def post_include(path, href):
    """Unwrap one include. Returns 'done' or 'already'."""
    text = read(path)
    m = wrapped_re(href).search(text)
    if m is None:
        if inc_re(href).search(text):
            return "already"
        sys.exit("include %s not found in %s" % (href, path))
    write(path, text[: m.start()] + m.group(2) + text[m.end():])
    return "done"


def files_referencing(ref_id):
    hits = []
    for name in os.listdir(SRC):
        if not name.endswith(".ptx"):
            continue
        if re.search(r'<xref\b[^>]*\bref="[^"]*\b%s\b[^"]*"' % re.escape(ref_id), read(os.path.join(SRC, name))):
            hits.append(name)
    return hits


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("post-notes", "unpost-notes"):
        sys.exit(__doc__)
    action, key = sys.argv[1], sys.argv[2]
    if key not in TOPICS:
        sys.exit("unknown topic key: %s" % key)
    path, href = TOPICS[key]

    originals = {p: read(p) for p in
                 [os.path.join(SRC, n) for n in os.listdir(SRC) if n.endswith(".ptx")]}

    def bail(msg):
        for p, s in originals.items():
            write(p, s)
        print(msg, file=sys.stderr)
        sys.exit(2)

    baseline = dangling_refs()

    if action == "post-notes":
        state = post_include(path, href)
        # restore any dependents that were unposted together with this topic
        pat = re.compile(
            r'<!--\s*UNPOSTED-WITH topic="%s"\s+(<xi:include\s+href="[^"]+"\s*/>)\s+UNPOSTED\s*-->' % re.escape(key)
        )
        for p in originals:
            text = read(p)
            new = pat.sub(lambda m: m.group(1), text)
            if new != text:
                write(p, new)
        print("post-notes %s: %s" % (key, state))
    else:
        state = unpost_include(path, href)
        if state == "done":
            # wrap dependent exercise sets until nothing dangles
            for _ in range(10):
                new_dangling = dangling_refs() - baseline
                if not new_dangling:
                    break
                fixed_any = False
                # anything main.ptx includes (a whole chapter, the whole
                # exercises chapter) is never an acceptable dependency
                # casualty of unposting one topic
                top_level = set(re.findall(r'href="\.?/?([^"]+)"', read(os.path.join(SRC, "main.ptx"))))
                # another topic's own file is never a casualty either:
                # taking a different topic off the site is an editorial
                # decision, not a mechanical one — bail to the Claude path
                topic_files = {h for _, h in TOPICS.values()}
                for ref in sorted(new_dangling):
                    for fname in files_referencing(ref):
                        fpath = os.path.join(SRC, fname)
                        if fpath in (path,) or fname in top_level or fname in topic_files:
                            continue
                        # only fix by wrapping the referencing file's own include
                        for holder in originals:
                            text = read(holder)
                            if inc_re(fname).search(text) or wrapped_re(fname).search(text):
                                if unpost_include(holder, fname, topic=key) == "done":
                                    fixed_any = True
                                break
                if not fixed_any:
                    bail("unpost %s leaves dangling cross-references (%s) that cannot be fixed by wrapping an include"
                         % (key, ", ".join(sorted(new_dangling))))
            else:
                bail("unpost %s: dependency fixing did not converge" % key)
        print("unpost-notes %s: %s" % (key, state))

    leftover = dangling_refs() - baseline
    if leftover:
        bail("change would leave dangling cross-references: %s" % ", ".join(sorted(leftover)))

    # well-formedness of every file we touched
    from xml.etree import ElementTree as ET
    for p, s in originals.items():
        if read(p) != s:
            try:
                ET.fromstring(read(p))
            except ET.ParseError as e:
                bail("%s no longer parses: %s" % (p, e))
            print("modified: %s" % p)


if __name__ == "__main__":
    main()
