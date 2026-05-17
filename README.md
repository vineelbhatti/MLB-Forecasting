# MLB Betting Model

A machine learning system for predicting MLB game outcomes and identifying edges against the betting market.

## What it does

- Predicts expected runs scored by each team using an XGBoost model trained on 4+ seasons of historical data
- Runs 10,000 Monte Carlo simulations per game to generate win, run-line, and total probabilities
- Compares model probabilities against live market odds to surface edges above a defined threshold
- Applies Kelly criterion to size bets, with a half-Kelly cap for risk management
- Scrapes Rotowire for late-breaking news (IL placements, pitcher scratches) and applies conservative post-prediction adjustments
- Tracks bet history, grades completed games, and shows P&L over time

## Stack

| Layer | Tech |
|---|---|
| Model | XGBoost (run prediction) + Negative Binomial Monte Carlo simulation |
| Data | pybaseball, MLB Stats API, The Odds API, OpenWeatherMap |
| News | Rotowire RSS feed |
| Backend | Python / Flask |
| Frontend | Vanilla JS, single-page app |

## Features

- **Dashboard** — live scores and real-time bet status for games you've taken
- **Today's Picks** — ranked edges with tier classification (Strong / Medium / Lean), odds, and unit recommendations
- **Game Projections** — full grid of today's games with win probabilities and run totals
- **History** — full graded bet log with win rate, P&L, ROI, and Brier score breakdowns by bet type and edge tier
- **News** — Rotowire beat reporter feed filtered to today's teams, with news-adjusted predictions shown inline on game cards

## Setup

```bash
# 1. Clone and create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r mlb-model/requirements.txt

# 3. Set API keys
cp mlb-model/.env.example mlb-model/.env
# Fill in ODDS_API_KEY and WEATHER_API_KEY

# 4. Run
cd mlb-model && python app.py
# Open http://localhost:5050
```

The model trains automatically on first run if no saved model is found. Re-training takes ~30 seconds.

## API Keys Required

| Key | Source | Used for |
|---|---|---|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | Live moneyline, spread, and totals odds |
| `WEATHER_API_KEY` | [openweathermap.org](https://openweathermap.org) | Wind and temperature features |

## Model

The run prediction model uses team-level season-to-date features:

- Batting: wOBA, OBP, SLG, K%, BB%, ISO, BABIP, wRC+
- Pitching: ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9
- Starting pitcher stats for the scheduled starter
- Weather: wind speed/direction, temperature
- Home/away indicator

Predictions feed into a Negative Binomial distribution parameterized per team, then simulated to produce full game probability distributions. A separate news adjustment layer applies small additive corrections for high-confidence late-breaking signals (pitcher scratches, IL placements) that are unlikely to be reflected in the season-to-date stats.

## Bet Grading

Mark bets as taken via the UI. After games complete, click **Grade Today** to automatically resolve outcomes against final scores pulled from the MLB Stats API. Results are logged to `logs/recommendations.csv`.
