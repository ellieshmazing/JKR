# JKR tweet report

A small personal project that collects public @jk_rowling posts, classifies them with an LLM
(TypeSafe Jev), and builds a static HTML report of the posts flagged as hostile toward trans people.

## Layout

- `docs/` – the generated reports
  - `jkr_report.html` – self-contained, images embedded (works offline)
  - `jkr_report_light.html` – small, images linked from X (needs internet)
- `src/` – pipeline scripts
  - `collect_jkr.py` – pull posts via the Sorsa API
  - `classify_jkr.py` – two-stage classification (gate, then topics); questions in `questions.py`
  - `highlight_spans.py` – pick out the offending spans in each post
  - `build_report.py` – render the HTML report
- `data/` – local working data (git-ignored; created by the scripts)

## Running it

Requires Python 3 and `pip install -r requirements.txt`, plus your own API keys:

```bash
export SORSA_API_KEY=...
export TYPESAFE_API_KEY=...
python src/collect_jkr.py
python src/classify_jkr.py --help
python src/highlight_spans.py --help
python src/build_report.py
```

Tweet content belongs to its authors; the data is not redistributed here.
