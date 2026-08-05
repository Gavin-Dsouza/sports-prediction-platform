const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Team {
  id: string;
  name: string;
  abbreviation: string;
}

export interface Game {
  id: string;
  game_date: string;
  game_datetime: string | null;
  status: string;
  home_team: Team;
  away_team: Team;
  home_score: number | null;
  away_score: number | null;
  venue_name: string | null;
}

export interface GamePrediction {
  game: Game;
  ensemble_home_win_probability: number | null;
  per_model: Record<string, number>;
  model_version: string | null;
}

export interface BetRecommendation {
  id: string;
  game_id: string;
  market: string;
  selection: string;
  predicted_probability: number;
  market_implied_probability: number;
  edge: number;
  expected_value: number;
  kelly_fraction: number;
  recommended_stake_fraction: number;
  confidence_score: number;
  rank: number | null;
  generated_at: string;
}

export interface BacktestResult {
  market: string;
  num_bets: number;
  accuracy: number | null;
  log_loss: number | null;
  brier_score: number | null;
  roi: number | null;
  max_drawdown: number | null;
  sharpe_ratio: number | null;
  max_losing_streak: number | null;
  calibration_curve: { buckets: Array<Record<string, number>> };
}

export interface BacktestRun {
  id: string;
  sport: string;
  predictor_name: string;
  model_version: string;
  start_date: string;
  end_date: string;
  notes: string | null;
  created_at: string;
  results: BacktestResult[];
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${path}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  games: (on?: string) => fetchJson<Game[]>(`/games${on ? `?on=${on}` : ""}`),
  predictionsToday: (on?: string) =>
    fetchJson<GamePrediction[]>(`/predictions/today${on ? `?on=${on}` : ""}`),
  recommendedBets: (on?: string) =>
    fetchJson<BetRecommendation[]>(`/bets/recommended${on ? `?on=${on}` : ""}`),
  backtests: () => fetchJson<BacktestRun[]>("/backtests"),
};
