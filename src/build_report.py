#!/usr/bin/env python3
"""Build a single self-contained HTML scroll of @jk_rowling tweets flagged as hostile to trans people.

Selection (hybrid, confidence-gated):
  - overall_score >= OVERALL_MIN and overall_conf >= OVERALL_CONF, OR
  - any single topic score >= TOPIC_MIN with that topic's conf >= TOPIC_CONF.
Order: overall_score descending. No model scores are shown in the output.

Parent tweets of replies are taken from the raw data when present, else fetched once (batched,
100 per request) from Sorsa and cached in parents.jsonl. Anything unavailable is skipped.

  python build_report.py                 # fetch missing parents (if key + credits) and build
  python build_report.py --no-fetch      # build from cache only
Output: docs/jkr_report.html (+ _light variant)
"""
import argparse, base64, csv, glob, hashlib, html, json, mimetypes, os, re, sys
from datetime import datetime
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CLASSIFIED = os.path.join(DATA, "jkr_classified.csv")
RAW = os.path.join(DATA, "jkr_tweets_raw.jsonl")
PARENTS = os.path.join(DATA, "parents.jsonl")
OUT = os.path.join(os.path.dirname(HERE), "docs", "jkr_report.html")
BULK = "https://api.sorsa.io/v3/tweet-info-bulk"

OVERALL_MIN, OVERALL_CONF = 2.5, 0.30
TOPIC_MIN, TOPIC_CONF = 3.0, 0.40
TOPICS = ["sex_gender_identity", "single_sex_spaces", "safety_and_predation", "sport",
          "youth_and_healthcare", "law_and_policy", "language_and_misgendering",
          "targeting_of_individuals_and_groups"]


def num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def select():
    rows = []
    for r in csv.DictReader(open(CLASSIFIED, encoding="utf-8-sig")):
        if r["gated_in"] != "True":
            continue
        o, oc = num(r["overall_score"]), num(r["overall_conf"])
        hit = o is not None and oc is not None and o >= OVERALL_MIN and oc >= OVERALL_CONF
        if not hit:
            for t in TOPICS:
                s, c = num(r[t + "_score"]), num(r[t + "_conf"])
                if s is not None and c is not None and s >= TOPIC_MIN and c >= TOPIC_CONF:
                    hit = True
                    break
        if hit:
            rows.append((o or 0.0, r["id"]))
    rows.sort(key=lambda x: -x[0])
    return rows


def load_jsonl(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                t = json.loads(line)
                out[str(t["id"])] = t
            except Exception:
                pass
    return out


def load_key():
    k = os.environ.get("SORSA_API_KEY")
    if k:
        return k.strip()
    return None


def fetch_parents(ids, key):
    ids = list(ids)
    with open(PARENTS, "a", encoding="utf-8") as f:
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            try:
                r = requests.post(BULK, json={"tweet_links": chunk},
                                  headers={"ApiKey": key, "Content-Type": "application/json"}, timeout=90)
            except requests.RequestException as e:
                print(f"  parent fetch failed ({e}); continuing without"); return
            if r.status_code != 200:
                print(f"  parent fetch HTTP {r.status_code}: {r.text[:200]}; continuing without"); return
            got = r.json().get("tweets") or []
            for t in got:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"  fetched {len(got)}/{len(chunk)} parent tweets")


# ---------------------------------------------------------------- image embedding
IMG_CACHE = os.path.join(DATA, "img_cache")
EMBED = True   # False -> images are linked by URL instead of embedded
AVATARS = {}   # url -> css class index (each avatar is embedded once, in the stylesheet)


def data_uri(url):
    """Download (once, cached in img_cache/) and return a data: URI, or None if unavailable."""
    if not url or not EMBED:
        return url or None
    os.makedirs(IMG_CACHE, exist_ok=True)
    h = hashlib.sha1(url.encode()).hexdigest()
    hit = glob.glob(os.path.join(IMG_CACHE, h + ".*"))
    if not hit:
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            return None
        ct = (r.headers.get("content-type") or "").split(";")[0]
        if r.status_code != 200 or not ct.startswith("image/"):
            return None
        ext = {"image/jpeg": ".jpg"}.get(ct) or mimetypes.guess_extension(ct) or ".img"
        hit = [os.path.join(IMG_CACHE, h + ext)]
        open(hit[0], "wb").write(r.content)
    mime = mimetypes.guess_type(hit[0])[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(hit[0], "rb").read()).decode()


# ---------------------------------------------------------------- rendering
URL_RE = re.compile(r"(https?://\S+)|(@\w{1,15})|(#\w+)")


def rich(text):
    out, pos = [], 0
    for m in URL_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        tok = m.group(0)
        if m.group(1):
            trail = ""
            while tok and tok[-1] in ".,;:!?)":
                trail = tok[-1] + trail; tok = tok[:-1]
            out.append(f'<span class="lnk">{html.escape(tok)}</span>{html.escape(trail)}')
        else:
            out.append(f'<span class="lnk">{html.escape(tok)}</span>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def rich_hl(text, spans):
    """rich() with <mark> around [s, e) spans; each line of a span is marked separately so line breaks stay unhighlighted."""
    if not spans:
        return rich(text)
    out, pos = [], 0
    for sp in spans:
        if sp["s"] < pos:
            continue
        out.append(rich(text[pos:sp["s"]]))
        for part in re.split(r"(\n+)", text[sp["s"]:sp["e"]]):
            if part.strip("\n"):
                out.append(f'<mark>{rich(part)}</mark>')
            else:
                out.append(part)
        pos = sp["e"]
    out.append(rich(text[pos:]))
    return "".join(out)


def when(s):
    for fmt in ("%a %b %d %H:%M:%S %z %Y",):
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%I:%M %p · %b %d, %Y").lstrip("0").replace(" 0", " ")
        except (ValueError, TypeError):
            pass
    return s or ""


def count(n):
    n = num(n)
    if n is None:
        return ""
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M".replace(".0M", "M")
    if n >= 10_000: return f"{n/1000:.0f}K"
    if n >= 1_000: return f"{n/1000:.1f}K".replace(".0K", "K")
    return str(n)


def avatar(user):
    url = (user.get("profile_image_url") or "").replace("_normal", "_bigger")
    if url and url not in AVATARS:
        uri = data_uri(url)
        AVATARS[url] = (len(AVATARS), uri)
    idx, uri = AVATARS.get(url, (None, None))
    if uri:
        return f'<div class="av a{idx}"></div>'
    name = user.get("display_name") or user.get("username") or "?"
    return f'<div class="av"><span>{html.escape(name[:1].upper())}</span></div>'


def media(t):
    bits = []
    for e in (t.get("entities") or []):
        if not isinstance(e, dict):
            continue
        if e.get("preview"):
            src = e["preview"]
        elif e.get("type") == "photo" and e.get("link"):
            src = e["link"] + ("&" if "?" in e["link"] else "?") + "name=small"
        else:
            continue
        uri = data_uri(src)
        if uri:
            tag = '<span class="play">▶</span>' if e.get("type") in ("video", "animated_gif") else ""
            bits.append(f'<div class="med"><img src="{html.escape(uri)}" alt="">{tag}</div>')
    return f'<div class="meds">{"".join(bits)}</div>' if bits else ""


def head(t):
    u = t.get("user") or {}
    badge = '<svg class="vb" viewBox="0 0 22 22"><circle cx="11" cy="11" r="10" fill="#1d9bf0"/><path fill="none" stroke="#fff" stroke-width="2" d="M6.500 11.500l3 3 6-6.500"/></svg>'         if u.get("verified") else ""
    return (f'<div class="hd"><b>{html.escape(u.get("display_name") or "")}</b>{badge}'
            f'<span class="mut">@{html.escape(u.get("username") or "")}</span></div>')


def quoted(q):
    if not isinstance(q, dict):
        return ""
    return (f'<div class="quote">{avatar(q.get("user") or {}).replace('class="av', 'class="av sm', 1)}'
            f'<div class="qb">{head(q)}<div class="tx">{rich(q.get("full_text") or "")}</div>{media(q)}</div></div>')


def parent_card(p):
    return (f'<div class="row parent"><div class="gut">{avatar(p.get("user") or {})}<div class="line"></div></div>'
            f'<div class="body">{head(p)}<div class="tx">{rich(p.get("full_text") or "")}</div>{media(p)}'
            f'<div class="mut small">{when(p.get("created_at"))}</div></div></div>')


def card(t, parent, spans=None):
    u = t.get("user") or {}
    stats = [("💬", t.get("reply_count")), ("🔁", t.get("retweet_count")), ("♥", t.get("likes_count")),
             ("👁", t.get("view_count"))]
    stats_html = "".join(f'<span><i>{ic}</i>{count(v)}</span>' for ic, v in stats if num(v) is not None)
    reply = ""
    if t.get("is_reply") and t.get("in_reply_to_username"):
        reply = f'<div class="mut small rep">Replying to <span class="lnk">@{html.escape(t["in_reply_to_username"])}</span></div>'
    url = f'https://x.com/{u.get("username", "jk_rowling")}/status/{t["id"]}'
    gut = f'<div class="gut">{avatar(u)}</div>'
    main = (f'<div class="row main">{gut}<div class="body">{head(t)}{reply}'
            f'<div class="tx big">{rich_hl(t.get("full_text") or "", spans)}</div>{media(t)}{quoted(t.get("quoted_status"))}'
            f'<div class="mut small ts">{when(t.get("created_at"))} · <a href="{url}" target="_blank" rel="noopener">View on X</a></div>'
            f'<div class="stats">{stats_html}</div></div></div>')
    return f'<article class="tw">{parent_card(parent) if parent else ""}{main}</article>'


CSS = """
:root{--bg:#000;--card:#16181c;--bd:#2f3336;--tx:#e7e9ea;--mut:#71767b;--blue:#1d9bf0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:620px;margin:0 auto;padding:16px 16px 80px}
.tw{background:var(--card);border:1px solid var(--bd);border-radius:16px;padding:12px 16px;margin:16px 0}
.row{display:flex;gap:12px}
.gut{display:flex;flex-direction:column;align-items:center;flex:none;width:40px}
.line{width:2px;flex:1;background:var(--bd);margin:4px 0 -4px}
.body{min-width:0;flex:1}
.parent .body{padding-bottom:14px}
.av{position:relative;width:40px;height:40px;border-radius:50%;overflow:hidden;background:#333639;flex:none;display:flex;align-items:center;justify-content:center;font-weight:700;color:#aaa}
.av{background-size:cover;background-position:center}
.av.sm{width:20px;height:20px;font-size:11px}
.hd{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.hd b{font-weight:700}.vb{width:18px;height:18px}
.mut{color:var(--mut)}.small{font-size:14px}
.rep{margin-top:2px}
.tx{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;margin-top:4px}
.tx.big{font-size:17px;margin-top:8px}
.lnk{color:var(--blue)}
.meds{display:grid;gap:2px;margin-top:10px;border-radius:14px;overflow:hidden;border:1px solid var(--bd);grid-template-columns:repeat(auto-fit,minmax(45%,1fr))}
.med{position:relative}.med img{display:block;width:100%;max-height:420px;object-fit:cover}
.play{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:34px;color:#fff;text-shadow:0 0 12px #000}
.quote{display:flex;gap:8px;margin-top:12px;border:1px solid var(--bd);border-radius:14px;padding:10px 12px}
.quote .qb{min-width:0;flex:1}
.ts{margin-top:12px;padding-bottom:10px;border-bottom:1px solid var(--bd)}
.ts a{color:var(--blue);text-decoration:none}.ts a:hover{text-decoration:underline}
.stats{display:flex;gap:22px;color:var(--mut);font-size:14px;padding-top:10px;flex-wrap:wrap}
.stats i{font-style:normal;margin-right:5px}
mark{background:rgba(249,24,128,.30);color:#fff;border-radius:4px;padding:1px 2px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
"""


def build(no_fetch, spans_path=None, out=None, embed=True):
    global EMBED
    EMBED = embed
    AVATARS.clear()
    sel = select()
    raw = load_jsonl(RAW)
    tweets = [(raw[i], sc) for sc, i in sel if i in raw]
    parents = load_jsonl(PARENTS)
    need = {str(t["in_reply_to_tweet_id"]) for t, _ in tweets
            if t.get("is_reply") and t.get("in_reply_to_tweet_id")} - set(raw) - set(parents)
    if need and not no_fetch:
        key = load_key()
        if key:
            print(f"Fetching {len(need)} parent tweets…")
            fetch_parents(need, key)
            parents = load_jsonl(PARENTS)
        else:
            print("No Sorsa key found; skipping parent fetch.")
    from highlight_spans import derive_spans, SPANS_OUT
    recs = load_jsonl(spans_path or SPANS_OUT)
    cards, with_parent, n_hl = [], 0, 0
    for t, _ in tweets:
        pid = str(t.get("in_reply_to_tweet_id") or "")
        p = raw.get(pid) or parents.get(pid) if t.get("is_reply") else None
        if p and not p.get("full_text"):
            p = None
        with_parent += bool(p)
        sp = derive_spans(t.get("full_text") or "", recs.get(str(t["id"])))
        n_hl += bool(sp)
        cards.append(card(t, p, sp))
    avcss = "".join(f'.a{i}{{background-image:url("{u}")}}' for i, u in AVATARS.values() if u)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Posts</title><style>{CSS}{avcss}</style></head><body>
<main>{"".join(cards)}</main>
</body></html>"""
    dest = out or OUT
    open(dest, "w", encoding="utf-8").write(page)
    print(f"Wrote {len(cards)} posts ({with_parent} with parent, {n_hl} with highlights) -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--spans", help="alternate results_spans.jsonl path")
    ap.add_argument("--out", help="alternate output html path")
    a = ap.parse_args()
    full = a.out or OUT
    light = os.path.splitext(full)[0] + "_light.html"
    build(a.no_fetch, a.spans, full, embed=True)    # images embedded: works offline
    build(True, a.spans, light, embed=False)         # images linked from X: small, needs internet
