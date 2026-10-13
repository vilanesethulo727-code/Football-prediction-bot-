# Football Prediction Lab (paper-trading MVP)

A starter Python app for comparing three football markets:
- 1X2 (home win / draw / away win)
- Over/under 2.5 total goals
- Both teams to score (BTTS)

It uses a transparent Poisson score model and a simple recent-form team-strength estimate. It is a baseline to test—not a proven edge and not a guaranteed-win system. It never places bets.

## 1) Install

Install Python 3.10 or newer. Extract this ZIP, open a terminal in the extracted folder, then run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Launch:
```bash
streamlit run app.py
```

Streamlit will show a local URL. On a computer, open it in your browser. To use it on an Android phone, the app must be hosted on a computer/server or cloud service and exposed securely; a local URL on your computer is not automatically accessible from your phone.

## 2) Prepare historical results

Upload a CSV with these exact column names:

`date,league,home_team,away_team,home_goals,away_goals`

Example:
```csv
date,league,home_team,away_team,home_goals,away_goals
2026-08-01,Premier League,Arsenal,Chelsea,2,1
2026-08-02,Premier League,Liverpool,Everton,1,1
2026-08-03,Premier League,Man City,Tottenham,3,0
2026-08-01,La Liga,Real Madrid,Sevilla,2,0
```

These are format examples only, not actual match results. Replace them with verified historical results. Use a reputable data provider whose terms permit your intended use. Football-Data.co.uk offers historical CSVs, but its terms state its free data is intended for private individuals and not commercial or data-training products using automated bots/scrapers/AI; check its current terms before using it for model training.

## 3) Prepare upcoming fixtures

Upload a CSV with these exact columns:

`date,league,home_team,away_team`

Example:
```csv
date,league,home_team,away_team
2026-10-13,Premier League,Arsenal,Liverpool
2026-10-13,La Liga,Barcelona,Atletico Madrid
```

Again, these are examples of the format, not verified fixtures. The competition name and team names must match the historical CSV. You can also enter a fixture manually in the app.

## 4) How to use it daily

1. Load updated historical results.
2. Load the next day's fixtures.
3. Review the three market probabilities.
4. Save predictions before kickoff. The app records a timestamp and probability snapshot.
5. After full time, enter the actual score.
6. Review market accuracy and download the CSV tracker.

The local SQLite database `paper_tracker.db` stores saved predictions. Keep a backup of that file.

## Important limitations

- Fixtures/results are currently supplied by you as CSV or manual entry. Automatic fixture feeds, injury news, lineups, and notifications are not connected in this first version.
- It uses the latest 20 home/away matches per team with shrinkage toward league averages; it does not yet model injuries, schedule congestion, xG, red cards, changing managers, or opponent-adjusted strength.
- Use chronological out-of-sample testing. Never evaluate the model on matches it was trained on.
- Track calibration and log loss/Brier score as well as hit rate. A high hit rate on short-priced favourites can still lose money at the odds available.
- No 90% win rate is promised. Do not use essential money or chase losses.
