#!/usr/bin/env python3
"""Pull @jk_rowling posts via the Sorsa API (free tier: ~100 requests, 20 posts each).

Strategy: page newest -> oldest through `from:jk_rowling` (order=latest) so every request
returns a full page of 20 and none are wasted on empty date windows. Progress is saved after
every request, so you can stop/resume freely. Run `python collect_jkr.py --help`.
"""
import argparse, csv, json, os, sys, time, glob
from datetime import datetime, timezone
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
API = "https://api.sorsa.io/v3/search-tweets"
RAW = os.path.join(DATA, "jkr_tweets_raw.jsonl")
STATE = os.path.join(DATA, "jkr_state.json")
CSV_OUT = os.path.join(DATA, "jkr_tweets.csv")
FIRST_RESPONSE = os.path.join(DATA, "first_response_sample.json")


def load_key():
    k = os.environ.get("SORSA_API_KEY")
    if k:
        return k.strip()
    sys.exit("No API key: set the SORSA_API_KEY environment variable.")


def pick(d, *names, default=None):
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return default


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {"requests_used": 0, "cursor": None, "until": None, "done": False}


def save_state(s):
    json.dump(s, open(STATE, "w", encoding="utf-8"), indent=1)


def seen_ids():
    ids = set()
    if os.path.exists(RAW):
        for line in open(RAW, encoding="utf-8"):
            try:
                t = json.loads(line)
                ids.add(str(pick(t, "id", "id_str", "tweet_id")))
            except Exception:
                pass
    return ids


def call(key, body):
    for attempt in range(5):
        try:
            r = requests.post(API, json=body, headers={"ApiKey": key, "Content-Type": "application/json"}, timeout=60)
        except requests.RequestException as e:
            print(f"  network error: {e}; retrying"); time.sleep(2 ** attempt); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  HTTP {r.status_code}; retrying"); time.sleep(2 ** attempt * 2); continue
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}  (401/403 = bad key, 402 = out of credits)")
    sys.exit("Gave up after repeated failures.")


def tweet_date(t):
    s = pick(t, "created_at", "date", "timestamp")
    if not s:
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect(a):
    key = load_key()
    st = load_state()
    ids = seen_ids()
    # remaining free-tier budget
    if a.reset:
        st = {"requests_used": 0, "cursor": None, "until": None, "done": False}
    if st["done"]:
        print("History already fully pulled. Use --since to pull newer posts, or --reset."); return
    print(f"Have {len(ids)} posts; {st['requests_used']}/{a.max_requests} requests used.")
    new_total = 0
    while st["requests_used"] < a.max_requests:
        q = "from:jk_rowling"
        if a.exclude_retweets:
            q += " -filter:nativeretweets"
        if a.exclude_replies:
            q += " -filter:replies"
        if a.since:
            q += f" since:{a.since}"
        if st["until"]:
            q += f" until:{st['until']}"
        body = {"query": q, "order": "latest"}
        if st["cursor"]:
            body["next_cursor"] = st["cursor"]
        data = call(load_key(), body)
        st["requests_used"] += 1
        if st["requests_used"] == 1 or not os.path.exists(FIRST_RESPONSE):
            json.dump(data, open(FIRST_RESPONSE, "w", encoding="utf-8"), indent=1)
        tweets = pick(data, "tweets", "data", "results", default=[]) or []
        cursor = pick(data, "next_cursor", "nextCursor", "cursor")
        fresh = [t for t in tweets if str(pick(t, "id", "id_str", "tweet_id")) not in ids]
        with open(RAW, "a", encoding="utf-8") as f:
            for t in fresh:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
                ids.add(str(pick(t, "id", "id_str", "tweet_id")))
        new_total += len(fresh)
        dates = [d for d in map(tweet_date, tweets) if d]
        oldest = min(dates).date() if dates else None
        print(f"  req {st['requests_used']:>3}: {len(tweets):>2} returned, {len(fresh):>2} new, oldest={oldest}, total={len(ids)}")
        if not tweets:
            st["done"] = True; st["cursor"] = None
        elif cursor and cursor != st["cursor"]:
            st["cursor"] = cursor
        elif oldest:  # no cursor offered: fall back to date windowing (until is exclusive; +1 day for same-day overlap)
            st["cursor"] = None
            nxt = oldest.isoformat()
            if st["until"] == nxt and not fresh:
                st["done"] = True
            st["until"] = nxt
        else:
            st["done"] = True
        save_state(st)
        if st["done"]:
            print("Reached the start of the available history."); break
        time.sleep(0.3)
    print(f"Done this run: {new_total} new posts. Requests used: {st['requests_used']}/{a.max_requests}.")
    export_csv()


def export_csv():
    if not os.path.exists(RAW):
        return
    rows, now = [], datetime.now(timezone.utc).isoformat()
    for line in open(RAW, encoding="utf-8"):
        t = json.loads(line)
        user = pick(t, "user", "author", default={}) or {}
        rows.append({
            "id": pick(t, "id", "id_str", "tweet_id"),
            "timestamp": (lambda d: d.isoformat() if d else pick(t, "created_at"))(tweet_date(t)),
            "url": f"https://x.com/jk_rowling/status/{pick(t, 'id', 'id_str', 'tweet_id')}",
            "is_reply": bool(pick(t, "is_reply", default=False)),
            "text": pick(t, "full_text", "text", default=""),
            "likes": pick(t, "likes_count", "favorite_count", "like_count"),
            "reposts": pick(t, "retweet_count", "retweets_count", "reposts_count"),
            "replies": pick(t, "reply_count", "replies_count"),
            "quotes": pick(t, "quote_count", "quotes_count"),
            "views": pick(t, "view_count", "views_count", "views"),
            "language": pick(t, "lang", "language"),
            "in_reply_to": pick(t, "in_reply_to_tweet_id", "in_reply_to_status_id", "in_reply_to_id"),
            "is_quote": bool(pick(t, "quoted_status", "quoted_tweet", "is_quote_status", default=False)),
            "username": pick(user, "username", "screen_name"),
            "collected_at": now,
        })
    rows.sort(key=lambda r: str(r["timestamp"]), reverse=True)
    rows = [r for r in rows if r["id"]]
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {CSV_OUT}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true", help="spend ONE request, save a sample response, and stop")
    p.add_argument("--max-requests", type=int, default=100, help="total request budget across all runs (default 100)")
    p.add_argument("--include-retweets", dest="exclude_retweets", action="store_false", help="keep native retweets")
    p.add_argument("--exclude-replies", action="store_true", help="skip replies (~85%% of her posts) so the budget reaches much further back")
    p.add_argument("--since", help="only posts on/after YYYY-MM-DD (use for later top-ups)")
    p.add_argument("--reset", action="store_true", help="forget saved progress (keeps raw file; duplicates are skipped)")
    p.add_argument("--export-only", action="store_true", help="just rebuild the CSV from the raw file")
    a = p.parse_args()
    if a.export_only:
        export_csv(); return
    if a.test:
        a.max_requests = 1 if a.reset else load_state()["requests_used"] + 1
    collect(a)


if __name__ == "__main__":
    main()
