"""
Monte Carlo simulation of MLB games using Negative Binomial distributions.
"""

import numpy as np
from scipy.stats import nbinom


N_SIMS = 10_000
RNG = np.random.default_rng(seed=42)


def _nb_params(mu: float, dispersion: float = 5.0) -> tuple[float, float]:
    """
    Convert mean (mu) to Negative Binomial (n, p) parameterization.
    Dispersion controls variance: var = mu + mu^2 / dispersion.
    Higher dispersion → closer to Poisson.
    """
    p = dispersion / (dispersion + mu)
    return dispersion, p


def simulate_game(
    home_runs_expected: float,
    away_runs_expected: float,
    n_sims: int = N_SIMS,
    dispersion: float = 5.0,
) -> dict:
    """
    Simulate a game N times using Negative Binomial run distributions.

    Parameters
    ----------
    home_runs_expected : float  — model predicted home team runs
    away_runs_expected : float  — model predicted away team runs
    n_sims : int                — number of Monte Carlo iterations
    dispersion : float          — NB dispersion parameter

    Returns
    -------
    dict with probabilities for win, cover, total outcomes
    """
    home_n, home_p = _nb_params(home_runs_expected, dispersion)
    away_n, away_p = _nb_params(away_runs_expected, dispersion)

    home_scores = RNG.negative_binomial(home_n, home_p, n_sims).astype(float)
    away_scores = RNG.negative_binomial(away_n, away_p, n_sims).astype(float)

    total_scores = home_scores + away_scores

    p_home_win = float(np.mean(home_scores > away_scores))
    p_away_win = float(np.mean(away_scores > home_scores))
    # ties go to extra innings — split evenly
    p_tie = 1.0 - p_home_win - p_away_win
    p_home_win += p_tie / 2
    p_away_win += p_tie / 2

    return {
        "home_runs_expected": round(home_runs_expected, 3),
        "away_runs_expected": round(away_runs_expected, 3),
        "total_expected": round(home_runs_expected + away_runs_expected, 3),
        "p_home_win": round(p_home_win, 4),
        "p_away_win": round(p_away_win, 4),
        # Run line: home -1.5 means home must win by 2+
        "p_home_cover_minus1_5": round(float(np.mean(home_scores - away_scores >= 1.5)), 4),
        "p_away_cover_plus1_5": round(float(np.mean(away_scores - home_scores >= -1.5)), 4),
        # Total: filled in dynamically per line via get_over_under_probs()
        "_home_scores": home_scores,
        "_away_scores": away_scores,
        "_total_scores": total_scores,
    }


def get_over_under_probs(sim_result: dict, total_line: float) -> dict:
    """
    Given simulation results and a book total line, return over/under probabilities.

    Parameters
    ----------
    sim_result  : dict returned by simulate_game()
    total_line  : float  — e.g. 8.5

    Returns
    -------
    dict with p_over and p_under
    """
    totals = sim_result.get("_total_scores", np.array([]))
    if len(totals) == 0:
        return {"p_over": 0.5, "p_under": 0.5, "total_line": total_line}
    p_over = float(np.mean(totals > total_line))
    p_under = float(np.mean(totals < total_line))
    p_push = 1.0 - p_over - p_under
    # Distribute push probability
    p_over += p_push / 2
    p_under += p_push / 2
    return {
        "total_line": total_line,
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
    }


def simulate_all_games(games_with_features: list[dict]) -> list[dict]:
    """
    Run simulations for a list of games.

    Each dict in games_with_features should have keys:
      game_pk, home_team, away_team, home_runs_expected, away_runs_expected, total_line (optional)

    Returns the input list augmented with simulation results.
    """
    results = []
    for game in games_with_features:
        sim = simulate_game(
            home_runs_expected=game["home_runs_expected"],
            away_runs_expected=game["away_runs_expected"],
        )
        game_result = {**game, **sim}

        total_line = game.get("total_line")
        if total_line is not None:
            ou = get_over_under_probs(sim, total_line)
            game_result.update(ou)

        # Remove internal numpy arrays before returning
        game_result.pop("_home_scores", None)
        game_result.pop("_away_scores", None)
        game_result.pop("_total_scores", None)

        results.append(game_result)
        print(
            f"  {game.get('away_team', '?')} @ {game.get('home_team', '?')}: "
            f"expected {sim['away_runs_expected']} - {sim['home_runs_expected']}, "
            f"home win {sim['p_home_win']:.1%}"
        )

    return results
