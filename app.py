import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "paper_tracker.db"
RESULTS_COLS = ["date", "league", "home_team", "away_team", "home_goals", "away_goals"]
FIXTURE_COLS = ["date", "league", "home_team", "away_team"]

st.set_page_config(page_title="Football Prediction Lab", page_icon="⚽", layout="wide")
st.title("⚽ Football Prediction Lab")
st.caption("Python Poisson-model starter • paper predictions only • no guaranteed outcomes")

st.warning(
    "This is a research and paper-tracking tool, not a guaranteed betting system. "
    "It does not place bets. Probabilities are estimates and must be tested against unseen matches."
)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                match_date TEXT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                lambda_home REAL,
                lambda_away REAL,
                p_home REAL,
                p_draw REAL,
                p_away REAL,
                p_over25 REAL,
                p_under25 REAL,
                p_btts_yes REAL,
                p_btts_no REAL,
                pick_1x2 TEXT,
                pick_goals TEXT,
                pick_btts TEXT,
                actual_home INTEGER,
                actual_away INTEGER,
                settled INTEGER DEFAULT 0
            )
        """)

init_db()

def normalize_results(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = set(RESULTS_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Historical results CSV is missing columns: {', '.join(sorted(missing))}")
    df = df[RESULTS_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=RESULTS_COLS)
    df["league"] = df["league"].astype(str)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    return df.sort_values("date")

def normalize_fixtures(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = set(FIXTURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Fixtures CSV is missing columns: {', '.join(sorted(missing))}")
    df = df[FIXTURE_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=FIXTURE_COLS)
    for c in ["league", "home_team", "away_team"]:
        df[c] = df[c].astype(str).str.strip()
    return df

def fit_league(results):
    # Simple, transparent team-strength model with shrinkage toward league averages.
    # Uses only the selected league and recent 20 matches per team.
    home_avg = float(results["home_goals"].mean())
    away_avg = float(results["away_goals"].mean())
    league_avg = max(0.2, (home_avg + away_avg) / 2)
    teams = sorted(set(results.home_team) | set(results.away_team))
    strengths = {}
    for team in teams:
        home_rows = results[results.home_team == team].tail(20)
        away_rows = results[results.away_team == team].tail(20)
        scored_home = home_rows.home_goals.sum()
        conceded_home = home_rows.away_goals.sum()
        scored_away = away_rows.away_goals.sum()
        conceded_away = away_rows.home_goals.sum()
        # Prior equivalent to 5 matches dampens tiny samples.
        home_n, away_n = len(home_rows), len(away_rows)
        attack_home = (scored_home + 5 * home_avg) / (home_n + 5) / max(home_avg, 0.2)
        defence_home = (conceded_home + 5 * away_avg) / (home_n + 5) / max(away_avg, 0.2)
        attack_away = (scored_away + 5 * away_avg) / (away_n + 5) / max(away_avg, 0.2)
        defence_away = (conceded_away + 5 * home_avg) / (away_n + 5) / max(home_avg, 0.2)
        strengths[team] = {
            "attack_home": float(np.clip(attack_home, 0.45, 1.8)),
            "defence_home": float(np.clip(defence_home, 0.45, 1.8)),
            "attack_away": float(np.clip(attack_away, 0.45, 1.8)),
            "defence_away": float(np.clip(defence_away, 0.45, 1.8)),
        }
    return {"home_avg": home_avg, "away_avg": away_avg, "teams": strengths, "league_avg": league_avg}

def predict(model, home, away):
    strengths = model["teams"]
    # Unknown teams fall back to neutral strength and are flagged in the UI.
    h = strengths.get(home, {"attack_home": 1.0, "defence_home": 1.0})
    a = strengths.get(away, {"attack_away": 1.0, "defence_away": 1.0})
    lam_h = float(np.clip(model["home_avg"] * h["attack_home"] * a.get("defence_away", 1.0), 0.15, 4.0))
    lam_a = float(np.clip(model["away_avg"] * a["attack_away"] * h.get("defence_home", 1.0), 0.15, 4.0))
    goals = np.arange(0, 11)
    ph = poisson.pmf(goals, lam_h)
    pa = poisson.pmf(goals, lam_a)
    matrix = np.outer(ph, pa)
    p_home = float(np.tril(matrix, -1).sum())  # row goals_home > column goals_away
    p_away = float(np.triu(matrix, 1).sum())
    p_draw = float(np.trace(matrix))
    p_over = float(sum(matrix[i, j] for i in range(11) for j in range(11) if i + j >= 3))
    p_btts = float(matrix[1:, 1:].sum())
    # Normalize 1X2 because score matrix is truncated at 10 goals.
    total = p_home + p_draw + p_away
    return {
        "lambda_home": lam_h, "lambda_away": lam_a,
        "p_home": p_home / total, "p_draw": p_draw / total, "p_away": p_away / total,
        "p_over25": p_over, "p_under25": 1 - p_over,
        "p_btts_yes": p_btts, "p_btts_no": 1 - p_btts
    }

def pct(x):
    return f"{100*x:.1f}%"

with st.sidebar:
    st.header("1. Load your data")
    results_file = st.file_uploader("Historical results CSV", type=["csv"])
    fixtures_file = st.file_uploader("Upcoming fixtures CSV", type=["csv"])
    st.markdown("**Data coverage:** add each league you want to analyse. The app does not fetch fixtures automatically yet.")
    st.markdown("**Required results columns:** `date, league, home_team, away_team, home_goals, away_goals`")
    st.markdown("**Required fixture columns:** `date, league, home_team, away_team`")

if not results_file:
    st.info("Upload a historical results CSV to train the first model. Use the sample CSVs included in the project ZIP to check the format.")
    st.subheader("What the bot will calculate")
    st.write("For each fixture: estimated home/draw/away probabilities, over/under 2.5 goals, both teams to score, and a transparent paper-tracking record.")
    st.stop()

try:
    results = normalize_results(pd.read_csv(results_file))
except Exception as e:
    st.error(str(e))
    st.stop()

if results.empty:
    st.error("No valid historical matches found in that CSV.")
    st.stop()

st.success(f"Loaded {len(results):,} historical matches across {results.league.nunique()} competition(s).")
leagues = sorted(results.league.unique().tolist())
league_filter = st.multiselect("Competitions to analyse", leagues, default=leagues)
results = results[results.league.isin(league_filter)]

if fixtures_file:
    try:
        fixtures = normalize_fixtures(pd.read_csv(fixtures_file))
        fixtures = fixtures[fixtures.league.isin(league_filter)]
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    st.subheader("2. Add a fixture manually")
    with st.form("manual_fixture"):
        c1, c2 = st.columns(2)
        with c1:
            match_date = st.date_input("Match date")
            league = st.selectbox("Competition", league_filter)
            home_team = st.text_input("Home team")
        with c2:
            away_team = st.text_input("Away team")
        submitted = st.form_submit_button("Add fixture to this session")
    if not submitted:
        st.stop()
    if not home_team.strip() or not away_team.strip() or home_team.strip() == away_team.strip():
        st.error("Enter two different team names.")
        st.stop()
    fixtures = pd.DataFrame([{"date": pd.Timestamp(match_date), "league": league, "home_team": home_team.strip(), "away_team": away_team.strip()}])

if fixtures.empty:
    st.info("No fixtures found for the selected competitions.")
    st.stop()

st.subheader("3. Today's paper predictions")
rows = []
for _, fx in fixtures.iterrows():
    league_results = results[results.league == fx.league].sort_values("date")
    if league_results.empty:
        continue
    model = fit_league(league_results)
    probs = predict(model, fx.home_team, fx.away_team)
    p1x2 = {"Home win": probs["p_home"], "Draw": probs["p_draw"], "Away win": probs["p_away"]}
    pick_1x2 = max(p1x2, key=p1x2.get)
    pick_goals = "Over 2.5" if probs["p_over25"] >= 0.5 else "Under 2.5"
    pick_btts = "Yes" if probs["p_btts_yes"] >= 0.5 else "No"
    known = fx.home_team in model["teams"] and fx.away_team in model["teams"]
    rows.append({
        "date": pd.Timestamp(fx.date).strftime("%Y-%m-%d"), "league": fx.league,
        "home_team": fx.home_team, "away_team": fx.away_team,
        **probs, "pick_1x2": pick_1x2, "pick_goals": pick_goals, "pick_btts": pick_btts,
        "known_teams": known
    })

if not rows:
    st.warning("No predictions could be made. Check that fixtures' league names exactly match the historical data.")
    st.stop()

preds = pd.DataFrame(rows)
display = preds[["date", "league", "home_team", "away_team", "p_home", "p_draw", "p_away", "p_over25", "p_under25", "p_btts_yes", "p_btts_no", "pick_1x2", "pick_goals", "pick_btts", "known_teams"]].copy()
for col in ["p_home", "p_draw", "p_away", "p_over25", "p_under25", "p_btts_yes", "p_btts_no"]:
    display[col] = display[col].map(pct)
display = display.rename(columns={
    "date":"Date","league":"Competition","home_team":"Home","away_team":"Away",
    "p_home":"Home %","p_draw":"Draw %","p_away":"Away %",
    "p_over25":"Over 2.5 %","p_under25":"Under 2.5 %",
    "p_btts_yes":"BTTS Yes %","p_btts_no":"BTTS No %",
    "pick_1x2":"Top 1X2 outcome","pick_goals":"Top goals pick","pick_btts":"Top BTTS pick",
    "known_teams":"Both teams in history?"
})
st.dataframe(display, use_container_width=True, hide_index=True)
st.caption("The top outcome in each market is simply the model's highest-probability side, not a recommendation to bet. Unknown teams use neutral strengths.")

if st.button("Save these predictions to paper tracker", type="primary"):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        for _, r in preds.iterrows():
            con.execute("""
                INSERT INTO predictions (
                    created_at, match_date, league, home_team, away_team,
                    lambda_home, lambda_away, p_home, p_draw, p_away,
                    p_over25, p_under25, p_btts_yes, p_btts_no,
                    pick_1x2, pick_goals, pick_btts
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now, r["date"], r["league"], r["home_team"], r["away_team"],
                r["lambda_home"], r["lambda_away"], r["p_home"], r["p_draw"], r["p_away"],
                r["p_over25"], r["p_under25"], r["p_btts_yes"], r["p_btts_no"],
                r["pick_1x2"], r["pick_goals"], r["pick_btts"]
            ))
    st.success(f"Saved {len(preds)} predictions. They are now locked into the local paper tracker.")

st.divider()
st.subheader("4. Settle completed matches")
st.write("Enter final scores for matches already saved. Use the same team names and competition.")
with sqlite3.connect(DB_PATH) as con:
    saved = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
if saved.empty:
    st.info("No saved predictions yet.")
else:
    pending = saved[saved.settled == 0].copy()
    if not pending.empty:
        pending["label"] = pending.apply(lambda r: f'#{r.id} | {r.match_date} | {r.league}: {r.home_team} vs {r.away_team}', axis=1)
        selected_label = st.selectbox("Choose a saved match", pending["label"].tolist())
        selected_id = int(pending.loc[pending.label == selected_label, "id"].iloc[0])
        with st.form("settle"):
            c1, c2 = st.columns(2)
            with c1:
                actual_home = st.number_input("Home goals", min_value=0, max_value=20, value=0, step=1)
            with c2:
                actual_away = st.number_input("Away goals", min_value=0, max_value=20, value=0, step=1)
            settle_submit = st.form_submit_button("Record final score")
        if settle_submit:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("UPDATE predictions SET actual_home=?, actual_away=?, settled=1 WHERE id=?",
                            (int(actual_home), int(actual_away), selected_id))
            st.success("Final score recorded.")
            st.rerun()
    else:
        st.success("All saved matches are settled.")

    settled = saved[saved.settled == 1].copy()
    if not settled.empty:
        def actual_1x2(r):
            if r.actual_home > r.actual_away: return "Home win"
            if r.actual_home < r.actual_away: return "Away win"
            return "Draw"
        settled["actual_1x2"] = settled.apply(actual_1x2, axis=1)
        settled["actual_goals"] = settled.apply(lambda r: "Over 2.5" if r.actual_home + r.actual_away >= 3 else "Under 2.5", axis=1)
        settled["actual_btts"] = settled.apply(lambda r: "Yes" if r.actual_home > 0 and r.actual_away > 0 else "No", axis=1)
        st.markdown("**Paper performance**")
        c1, c2, c3 = st.columns(3)
        c1.metric("1X2 pick accuracy", pct((settled.pick_1x2 == settled.actual_1x2).mean()))
        c2.metric("Over/under accuracy", pct((settled.pick_goals == settled.actual_goals).mean()))
        c3.metric("BTTS accuracy", pct((settled.pick_btts == settled.actual_btts).mean()))
        st.caption(f"Based on {len(settled)} settled match(es). Small samples are highly unreliable; compare over hundreds of matches.")
        st.dataframe(settled[["match_date","league","home_team","away_team","actual_home","actual_away","pick_1x2","actual_1x2","pick_goals","actual_goals","pick_btts","actual_btts"]], use_container_width=True, hide_index=True)

    st.download_button(
        "Download full paper tracker CSV",
        data=saved.to_csv(index=False).encode("utf-8"),
        file_name="football_paper_tracker.csv",
        mime="text/csv"
    )
