#!/usr/bin/env python3
"""Two-stage TypeSafe (Jev) classification of JKR tweets.

Stage 1 (gate):   the unchanged QUESTIONS from questions.py. A tweet passes if it looks like
                  gender politics or mentions trans people / gender identity (thresholds are
                  deliberately low: favour recall, so no transphobic tweet is gated out).
Stage 2 (topics): only for tweets that passed the gate. One `score` question per topic on a
                  0-4 scale (0 = topic not addressed ... 4 = demeaning/dehumanising), plus an overall score.

Results are appended to JSONL files keyed by tweet id, so runs resume and skip finished ids.
Key: env TYPESAFE_API_KEY only. Never printed, only ever sent to api.typesafe.ai.

  python classify_jkr.py gate --limit 20        # try a few first
  python classify_jkr.py gate                   # all tweets
  python classify_jkr.py topics                 # gated-in tweets only
  python classify_jkr.py export                 # merged CSV: jkr_classified.csv
  python classify_jkr.py sample 150             # label_sheet.csv for hand-labelling
  python classify_jkr.py eval                   # gate precision/recall vs. label_sheet.csv
"""
import argparse, csv, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from questions import QUESTIONS as GATE_QUESTIONS  # reused unchanged so results stay comparable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"  # pinned (jev-latest resolved to this) so thresholds stay valid
TWEETS = os.path.join(DATA, "jkr_tweets.csv")
GATE_OUT = os.path.join(DATA, "results_gate.jsonl")
TOPIC_OUT = os.path.join(DATA, "results_topics.jsonl")
EXPORT = os.path.join(DATA, "jkr_classified.csv")
LABELS = os.path.join(DATA, "label_sheet.csv")

# Gate defaults; tune with `eval` on hand-labelled data (0.5 is not assumed right).
GATE_GENDER_P = 0.20   # P(category == gender_politics)
GATE_TRANS_P = 0.25    # P(mentions trans people / gender identity)

# --- Stage 2 ------------------------------------------------------------------------------
# Same ordered levels for every topic so scores are comparable across topics.
LEVELS = [
    "The post does not address this topic at all.",
    "The post addresses this topic without hostility toward trans people: neutral, factual, or supportive of trans people or trans-inclusive positions.",
    "The post argues a gender-critical position on this topic in measured terms, with no demeaning language, generalisation or accusation aimed at trans people.",
    "The post denies trans people's identities or argues for their exclusion on this topic, and relies on negative generalisations or stereotypes about trans people.",
    "The post is demeaning, dehumanising, mocking or accusatory toward trans people on this topic (e.g. portraying them as predators or a threat, ridicule, misgendering as an insult).",
]

TOPICS = {
    "sex_gender_identity": "the nature of sex and gender identity: who counts as a woman or man, whether trans women are women, whether trans identities are valid or real",
    "single_sex_spaces": "single-sex spaces such as toilets, changing rooms, shelters, hospitals and who may use them",
    "safety_and_predation": "safety, violence, sexual offending or risk to women and girls in connection with trans people (including prisons)",
    "sport": "sport and competition categories, and trans athletes",
    "youth_and_healthcare": "children and young people, puberty blockers, medical or social transition, gender clinics, schools",
    "law_and_policy": "law and policy: the Equality Act, Gender Recognition Act, self-ID, court rulings, legislation, organisational policy on sex and gender",
    "language_and_misgendering": "how trans people are named and described: pronouns, deadnaming, misgendering, slurs, euphemisms, or mocking terminology",
    "targeting_of_individuals_and_groups": "named trans people, trans activists, supporters of trans rights, or organisations, who are singled out for criticism, mockery or attack",
}
OVERALL = "all topics taken together"


def _topic_question(desc):
    return {
        "type": "score",
        "instructions": (
            f"How transphobic is the post with respect to {desc}? "
            "Judge only what this post itself says, not the poster's wider reputation. "
            "Choose the level that matches the most hostile statement the post makes on this topic."
        ),
        "criteria": LEVELS,
    }


TOPIC_QUESTIONS = {k: _topic_question(v) for k, v in TOPICS.items()}
TOPIC_QUESTIONS["overall"] = _topic_question(OVERALL)
# A score is a probability-weighted position, so it blurs when a topic is absent. A separate presence
# check lets code ignore scores for topics the post does not touch.
for _k, _v in TOPICS.items():
    TOPIC_QUESTIONS[f"{_k}_present"] = {"type": "noul", "instructions": f"Does the post say anything about {_v}?"}
SCORE_KEYS = [k for k in TOPIC_QUESTIONS if not k.endswith("_present")]
PRESENT_MIN = 0.5  # presence threshold used by export


# --- API ----------------------------------------------------------------------------------
def call(key, model, text, questions, max_tries=6):
    body = {"model": model, "state": {"post": text}, "questions": questions}
    for attempt in range(max_tries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json=body, timeout=60)
        except requests.RequestException as e:
            if attempt == max_tries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 422):
            sys.exit(f"HTTP {r.status_code} (fatal): {r.text[:400]}")
        if r.status_code in (429, 529) or r.status_code >= 500:
            time.sleep(2 ** attempt + random.random())
            continue
        sys.exit(f"HTTP {r.status_code}: {r.text[:400]}")
    raise RuntimeError("gave up after retries")


# --- IO helpers ---------------------------------------------------------------------------
def read_tweets():
    with open(TWEETS, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["text"].strip()]


def read_jsonl(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    out[d["id"]] = d
    return out


def gate_passes(answers, gender_p=GATE_GENDER_P, trans_p=GATE_TRANS_P):
    gp = answers["category"]["probabilities"].get("gender_politics", 0.0)
    return gp >= gender_p or answers["mentions_trans_people"]["noul"] >= trans_p


def run(rows, out_path, questions, model, workers, key):
    lock = threading.Lock()
    done = 0

    def work(row):
        data = call(key, model, row["text"], questions)
        return row["id"], data

    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for fut in as_completed(futs):
            tid, data = fut.result()
            rec = {"id": tid, "model": data.get("model"), "answers": data["answers"], "usage": data.get("usage")}
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done += 1
                if done % 25 == 0 or done == len(rows):
                    print(f"  {done}/{len(rows)}", flush=True)


def get_key():
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        sys.exit("Set TYPESAFE_API_KEY first.")
    return key


# --- Commands -----------------------------------------------------------------------------
def cmd_gate(a):
    done = read_jsonl(GATE_OUT)
    rows = [r for r in read_tweets() if r["id"] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"gate: {len(rows)} tweets to do ({len(done)} already done), model={a.model}")
    if a.dry_run or not rows:
        return
    run(rows, GATE_OUT, GATE_QUESTIONS, a.model, a.workers, get_key())


def cmd_topics(a):
    gate = read_jsonl(GATE_OUT)
    done = read_jsonl(TOPIC_OUT)
    tweets = {r["id"]: r for r in read_tweets()}
    rows = [tweets[i] for i, g in gate.items()
            if i in tweets and i not in done and gate_passes(g["answers"], a.gender_p, a.trans_p)]
    if a.limit:
        rows = rows[: a.limit]
    print(f"topics: {len(rows)} gated-in tweets to do ({len(done)} already done), model={a.model}")
    if a.dry_run or not rows:
        return
    run(rows, TOPIC_OUT, TOPIC_QUESTIONS, a.model, a.workers, get_key())


def cmd_export(a):
    gate, topics = read_jsonl(GATE_OUT), read_jsonl(TOPIC_OUT)
    cols = ["id", "timestamp", "url", "text", "likes", "views", "gate_gender_p", "gate_mentions_trans", "gate_hostile",
            "gated_in"] + [f"{t}_present" for t in TOPICS] + [f"{t}_score" for t in SCORE_KEYS] + [f"{t}_conf" for t in SCORE_KEYS]
    with open(EXPORT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in read_tweets():
            g = gate.get(r["id"])
            if not g:
                continue
            ga = g["answers"]
            row = {c: r.get(c, "") for c in ("id", "timestamp", "url", "text", "likes", "views")}
            row.update(gate_gender_p=ga["category"]["probabilities"].get("gender_politics"),
                       gate_mentions_trans=ga["mentions_trans_people"]["noul"], gate_hostile=ga["hostile"]["noul"],
                       gated_in=gate_passes(ga, a.gender_p, a.trans_p))
            ta = topics.get(r["id"], {}).get("answers", {})
            for t in TOPICS:
                if t in ta:
                    row[f"{t}_present"] = ta[f"{t}_present"]["noul"]
            for t in SCORE_KEYS:
                if t not in ta:
                    continue
                # blank the score when the topic isn't present; "overall" is always kept
                if t == "overall" or ta[f"{t}_present"]["noul"] >= PRESENT_MIN:
                    row[f"{t}_score"] = ta[t].get("score")
                    row[f"{t}_conf"] = ta[t].get("confidence")
            w.writerow(row)
    print(f"wrote {EXPORT}")


def cmd_sample(a):
    gate = read_jsonl(GATE_OUT)
    rows = [r for r in read_tweets() if r["id"] in gate]
    random.Random(0).shuffle(rows)
    with open(LABELS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "human_gate"])  # human_gate: 1 = gender politics / trans-related, 0 = not
        for r in rows[: a.n]:
            w.writerow([r["id"], r["text"], ""])
    print(f"wrote {LABELS}: fill in human_gate (1/0), then run `eval`")


def cmd_eval(a):
    gate = read_jsonl(GATE_OUT)
    with open(LABELS, encoding="utf-8-sig", newline="") as f:
        lab = {r["id"]: r["human_gate"].strip() for r in csv.DictReader(f) if r["human_gate"].strip() in ("0", "1")}
    if not lab:
        sys.exit("No labels filled in.")
    print(f"{len(lab)} labelled; {sum(v == '1' for v in lab.values())} positive")
    print("gender_p trans_p  precision  recall")
    for gp in (0.1, 0.2, 0.3, 0.5):
        for tp in (0.15, 0.25, 0.4, 0.5):
            tpc = fp = fn = 0
            for i, v in lab.items():
                if i not in gate:
                    continue
                p = gate_passes(gate[i]["answers"], gp, tp)
                tpc += p and v == "1"; fp += p and v == "0"; fn += (not p) and v == "1"
            prec = tpc / (tpc + fp) if tpc + fp else float("nan")
            rec = tpc / (tpc + fn) if tpc + fn else float("nan")
            print(f"{gp:8.2f} {tp:7.2f}  {prec:9.2f}  {rec:6.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("gate", cmd_gate), ("topics", cmd_topics), ("export", cmd_export)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--gender-p", type=float, default=GATE_GENDER_P)
        s.add_argument("--trans-p", type=float, default=GATE_TRANS_P)
        if name != "export":
            s.add_argument("--limit", type=int)
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--model", default=DEFAULT_MODEL)
            s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("sample"); s.set_defaults(fn=cmd_sample); s.add_argument("n", type=int, nargs="?", default=150)
    s = sub.add_parser("eval"); s.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
