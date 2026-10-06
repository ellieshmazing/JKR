#!/usr/bin/env python3
"""Find the most offensive passages inside each flagged tweet, for highlighting in the report.

Approach (hierarchical, context-aware, resumable):
  1. Split each selected tweet into sentences (character offsets kept).
  2. Ask Jev about every sentence. The state holds the FULL post plus the marked PASSAGE, so the
     model can resolve "they"/"these people" but is told to judge only the passage:
        group_ref  (noul)  does the passage refer to trans women / trans people as a group?
        hostility  (score) 0-4, same ladder as the post-level analysis, applied to the passage.
  3. Sentences that look plausibly hostile are split into clauses (commas, dashes, colons,
     brackets) and the clauses are scored the same way, so the highlight can be a tight phrase.
  4. Raw scores for every scored segment are stored. Which segments get highlighted is decided
     later by derive_spans(), a pure function, so thresholds can be tuned WITHOUT re-running the model.

Highlight = passage that is hostile AND refers to trans women/people as a group.

  python highlight_spans.py run --dry-run      # count tweets/sentences, no API calls
  python highlight_spans.py run --limit 5      # try a few
  python highlight_spans.py run                # all selected tweets (resumes; skips finished ids)
  python highlight_spans.py show 15            # print highlighted passages for a quick sanity check
Key: env TYPESAFE_API_KEY (same as classify_jkr.py).
"""
import argparse, hashlib, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import classify_jkr as cj          # reuses call(), get_key(), DEFAULT_MODEL
import build_report as br          # reuses select(), load_jsonl() for the same tweet set

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SPANS_OUT = os.path.join(DATA, "results_spans.jsonl")

# Highlight decision thresholds (applied at build time, not run time)
GROUP_MIN = 0.50      # group_ref probability to count as "refers to a group"
HOST_MIN = 3.0        # hostility needed for a group-directed passage
# Which sentences get split into clauses at run time (deliberately looser than the above)
REFINE_HOST = 2.5
REFINE_GROUP = 0.30
REFINE_MIN_WORDS = 8

LEVELS = [
    "The passage does not concern trans people at all.",
    "The passage concerns trans people without hostility: neutral, factual, or supportive of trans people or trans-inclusive positions.",
    "The passage argues a gender-critical position in measured terms, with no demeaning language, generalisation or accusation aimed at trans people.",
    "The passage denies trans people's identities or argues for their exclusion, and relies on negative generalisations or stereotypes about trans people.",
    "The passage is demeaning, dehumanising, mocking or accusatory toward trans people (e.g. portraying them as predators or a threat, ridicule, misgendering as an insult).",
]
QUESTIONS = {
    "group_ref": {
        "type": "noul",
        "instructions": (
            "The state holds a full post and, after the marker, one PASSAGE taken from it. Judge only the PASSAGE; use the rest "
            "of the post only to understand what it refers to. Does the PASSAGE refer to trans women, or trans people in general, "
            "as a group or category (for example 'trans women', 'transwomen', 'men who identify as women', 'males', or a pronoun "
            "such as 'they' that clearly stands for such a group)? Answer no if it concerns only one named individual or does not "
            "concern trans people."
        ),
    },
    "hostility": {
        "type": "score",
        "instructions": (
            "The state holds a full post and, after the marker, one PASSAGE taken from it. Judge only the PASSAGE; use the rest "
            "of the post only to understand what it refers to. How hostile is the PASSAGE toward trans people? "
            "Choose the level matching the most hostile thing the passage itself says."
        ),
        "criteria": LEVELS,
    },
}


# ------------------------------------------------------------------ segmentation
SENT_SPLIT = re.compile(r"""(?:(?<=[.!?])|(?<=[.!?]["')\]”’]))\s+|\s*\n+\s*""")
CLAUSE_SPLIT = re.compile(r"\s*[;:—–]\s*|,\s+|\s+-\s+|\s*[()]\s*")
URL = re.compile(r"https?://\S+")
LEAD_MENTIONS = re.compile(r"^(?:@\w+\s+)+")


def _pieces(text, pattern, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    pos = lo
    for m in pattern.finditer(text, lo, hi):
        yield pos, m.start()
        pos = m.end()
    yield pos, hi


def _trim(text, s, e):
    """Drop whitespace, leading @mentions and leading/trailing stray punctuation from [s, e)."""
    seg = text[s:e]
    m = LEAD_MENTIONS.match(seg)
    if m:
        s += m.end()
    while s < e and (text[s].isspace() or text[s] in ",;:—–-)("):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in ",;:—–-("):
        e -= 1
    return s, e


def words(text, s, e):
    return len(re.findall(r"[A-Za-z']{2,}", URL.sub(" ", text[s:e])))


def sentences(text):
    out = []
    for s, e in _pieces(text, SENT_SPLIT):
        s, e = _trim(text, s, e)
        if e > s and words(text, s, e) >= 3:
            out.append((s, e))
    return out


def clauses(text, s, e):
    """Clause chunks of [s, e); chunks under 4 words are merged into a neighbour so each is a real phrase."""
    chunks = [_trim(text, a, b) for a, b in _pieces(text, CLAUSE_SPLIT, s, e)]
    chunks = [(a, b) for a, b in chunks if b > a]
    merged = []
    for a, b in chunks:
        if merged and (words(text, a, b) < 4 or words(text, *merged[-1]) < 4):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


# ------------------------------------------------------------------ model calls
def state_text(full, s, e):
    return f"FULL POST:\n{full}\n\nPASSAGE (judge only this part):\n{full[s:e]}"


def score_segment(key, model, full, s, e):
    a = cj.call(key, model, state_text(full, s, e), QUESTIONS)["answers"]
    return {"gr": a["group_ref"]["noul"], "host": a["hostility"].get("score"), "conf": a["hostility"].get("confidence")}


def refine_wanted(text, s, e, r):
    host = r["host"] or 0
    return words(text, s, e) >= REFINE_MIN_WORDS and host >= REFINE_HOST and r["gr"] >= REFINE_GROUP


def process(key, model, tid, text):
    segs = []
    for s, e in sentences(text):
        r = score_segment(key, model, text, s, e)
        segs.append({"s": s, "e": e, "kind": "sent", "parent": None, **r})
        if refine_wanted(text, s, e, r):
            idx = len(segs) - 1
            cl = clauses(text, s, e)
            if len(cl) >= 2:
                for cs, ce in cl:
                    segs.append({"s": cs, "e": ce, "kind": "clause", "parent": idx,
                                 **score_segment(key, model, text, cs, ce)})
    return {"id": tid, "sha": sha(text), "model": model, "segments": segs}


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ highlight decision (pure)
def tier(seg, group_min=GROUP_MIN, host_min=HOST_MIN):
    return 1 if seg["gr"] >= group_min and (seg.get("host") or 0) >= host_min else 0


def derive_spans(text, rec, **kw):
    """-> sorted list of {'s','e','tier'}; [] if the record doesn't match this text."""
    if not rec or rec.get("sha") != sha(text):
        return []
    segs = rec["segments"]
    kids = {}
    for i, sg in enumerate(segs):
        if sg["parent"] is not None:
            kids.setdefault(sg["parent"], []).append(sg)
    spans = []
    for i, sg in enumerate(segs):
        if sg["kind"] != "sent":
            continue
        ch = kids.get(i)
        picked = [(c, tier(c, **kw)) for c in ch if tier(c, **kw)] if ch else []
        if picked:
            spans += [{"s": c["s"], "e": c["e"], "tier": t} for c, t in picked]
        elif tier(sg, **kw):
            spans.append({"s": sg["s"], "e": sg["e"], "tier": tier(sg, **kw)})
    spans.sort(key=lambda x: x["s"])
    merged = []
    for sp in spans:  # join neighbours separated only by punctuation/spaces (never across a line break)
        if merged and merged[-1]["tier"] == sp["tier"] and not re.search(r"\w|\n", text[merged[-1]["e"]:sp["s"]]):
            merged[-1]["e"] = sp["e"]
        else:
            merged.append(dict(sp))
    return merged


# ------------------------------------------------------------------ commands
def targets():
    raw = br.load_jsonl(br.RAW)
    return [(i, raw[i]["full_text"]) for _, i in br.select() if i in raw and raw[i].get("full_text")]


def cmd_run(a):
    done = br.load_jsonl(SPANS_OUT)
    rows = [(i, t) for i, t in targets() if not (i in done and done[i].get("sha") == sha(t))]
    if a.limit:
        rows = rows[: a.limit]
    n_sent = sum(len(sentences(t)) for _, t in rows)
    print(f"{len(rows)} tweets to do ({len(done)} already done); {n_sent} sentence calls + clause calls for hostile-looking sentences")
    if a.dry_run or not rows:
        return
    key, lock, n = cj.get_key(), threading.Lock(), 0
    with open(SPANS_OUT, "a", encoding="utf-8") as out, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, key, a.model, i, t) for i, t in rows]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()
                n += 1
                if n % 10 == 0 or n == len(rows):
                    print(f"  {n}/{len(rows)}", flush=True)


def cmd_show(a):
    recs = br.load_jsonl(SPANS_OUT)
    k = 0
    for i, t in targets():
        if i not in recs:
            continue
        sp = derive_spans(t, recs[i])
        k += 1
        print(f"\n[{i}]")
        for x in sp:
            print(f"  tier{x['tier']}: {t[x['s']:x['e']]!r}")
        if not sp:
            print("  (no highlight)")
        if k >= a.n:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run"); s.set_defaults(fn=cmd_run)
    s.add_argument("--limit", type=int); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--model", default=cj.DEFAULT_MODEL); s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("show"); s.set_defaults(fn=cmd_show); s.add_argument("n", type=int, nargs="?", default=15)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
