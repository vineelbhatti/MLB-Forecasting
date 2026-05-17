# MLB Betting Model

An end-to-end MLB betting model that pulls live data, builds team feature vectors, predicts runs with XGBoost, simulates games 10,000 times with Negative Binomial distributions, and surfaces bets with positive expected value vs. current book lines.

## Setup

### 1. Install dependencies

```bash
cd mlb-model
pip install -r requirements.txt
```

### 2. Configure API keys

Edit `.env` and add your keys:

```env
ODDS_API_KEY=your_key_from_the-odds-api.com
WEATHER_API_KEY=your_key_from_openweathermap.org
```

- **The Odds API**: free tier at [the-odds-api.com](https://the-odds-api.com) — 500 requests/month
- **OpenWeatherMap**: free tier at [openweathermap.org](https://openweathermap.org/api) — 1,000 calls/day

The MLB Stats API requires no key.

### 3. Run the model

```bash
cd src
python main.py
```

On first run, the model will train on synthetic data (or real historical data if pybaseball's Retrosheet game logs are available). The trained model is saved to `models/runs_model.joblib` and reused on subsequent runs.

## Pipeline Overview

```
data_pipeline.py  →  features.py  →  model.py  →  simulate.py  →  odds.py
      ↓                   ↓              ↓              ↓             ↓
  Raw CSVs           Feature          XGBoost        10,000x       Edge
 (data/raw/)         vectors          predict       NegBinom       table
```

### Data Sources

| Source | What it provides | Key |
|--------|-----------------|-----|
| pybaseball (FanGraphs) | Team wRC+, OBP, SLG, FIP, xFIP, BB%, K% | None |
| MLB Stats API | Today's games, starting pitchers, bullpen logs | None |
| The Odds API | Run lines and totals | ODDS_API_KEY |
| OpenWeatherMap | Wind speed/direction at ballpark | WEATHER_API_KEY |

### Feature Vector (per team per game)

| Feature | Description |
|---------|-------------|
| `starter_fip` | Starting pitcher FIP (season) |
| `starter_xfip` | Starting pitcher xFIP (season) |
| `bullpen_era` | Bullpen ERA (last 7 days) |
| `bullpen_pitches_7d` | Total bullpen pitches (last 7 days) |
| `wrc_plus` | Team wRC+ vs. opposing starter handedness |
| `obp` | Team OBP |
| `slg` | Team SLG |
| `park_factor` | Stadium park factor (runs, relative to 100) |
| `wind_speed` | Wind speed (mph) |
| `wind_encoded` | Wind component along home→CF axis (+ = out, − = in) |
| `ump_k_rate` | Home plate umpire strikeout rate |
| `pitcher_hand_L` | 1 if starter is left-handed |
| `opposing_hand_L` | 1 if opposing starter is left-handed |

### Simulation

Each game is simulated 10,000 times. Both teams' run totals are drawn from a **Negative Binomial** distribution parameterized by the XGBoost predicted mean. This correctly captures the overdispersion in MLB run scoring (more variance than Poisson).

Outputs per game:
- P(home win), P(away win)
- P(home covers -1.5), P(away covers +1.5)
- P(over N), P(under N) for the book total

### Edge Calculation

Book implied probabilities are extracted from American odds, vig-removed (fair odds), and compared to simulation probabilities:

```
edge = our_probability - book_fair_probability
```

Only bets with `edge > 3%` are surfaced.

## Output

### Console

```
Matchup                                  Away   Home  HWin%   Total
---------------------------------------- ------ ------ ------- -------
Yankees @ Red Sox                         4.2    4.8   54.3%    9.0

======================================================================
  TODAY'S BETTING RECOMMENDATIONS  (edge > 3%)
╭────┬─────────────────────────┬────────────┬──────────┬──────────┬───────┬───────╮
│    │ Game                    │ Bet        │ Our_Prob │ Book_Prob│ Edge  │ Odds  │
├────┼─────────────────────────┼────────────┼──────────┼──────────┼───────┼───────┤
│  0 │ Yankees @ Red Sox       │ Over 8.5   │ 58.4%    │ 52.4%    │ +6.0% │  -110 │
╰────┴─────────────────────────┴────────────┴──────────┴──────────┴───────┴───────╯
```

### Logs

All recommendations are appended to `logs/recommendations.csv`:

```
date,game,bet,our_prob,book_prob,edge,odds,result
2025-04-01,Yankees @ Red Sox,Over 8.5,58.4%,52.4%,+6.0%,-110,
```

Fill in the `result` column (W/L) after games finish to track performance.

## Project Structure

```
mlb-model/
├── .env                    # API keys (never commit this)
├── data/
│   ├── raw/                # Raw CSVs with today's date in filename
│   └── processed/          # (reserved for future use)
├── models/
│   └── runs_model.joblib   # Trained XGBoost model
├── logs/
│   └── recommendations.csv # Running bet log
├── src/
│   ├── data_pipeline.py    # Data fetching from all sources
│   ├── features.py         # Feature engineering
│   ├── model.py            # XGBoost training and prediction
│   ├── simulate.py         # Monte Carlo simulation
│   ├── odds.py             # Edge calculation and output formatting
│   └── main.py             # Pipeline orchestrator
└── requirements.txt
```

## Retraining the Model

Delete `models/runs_model.joblib` and rerun `main.py`. The model trains automatically. To force a retrain with more seasons, call directly:

```python
from model import train_model
train_model(seasons=5)
```

## Notes

- **Historical training data**: pybaseball's Retrosheet game logs provide team-level historical results. If unavailable, the model falls back to 20,000 synthetic samples generated from realistic MLB distributions.
- **Park factors**: Hardcoded per-stadium values. Update annually.
- **Umpire data**: Uses league average K rate by default. Wire in a real umpire database (e.g., Umpire Scorecards API) for improvement.
- **Handedness splits**: FanGraphs team-level data doesn't always include L/R splits. The feature uses overall wRC+ when splits aren't available.
