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

export interface FeatureReason {
  feature: string;
  contribution: number | null;
  importance: number | null;
}

export interface SimilarGame {
  game_id: string;
  similarity: number;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
}

export interface PredictionExplanation {
  top_reasons: FeatureReason[];
  also_considered: FeatureReason[];
  shap_model_weight: number;
  similar_games: SimilarGame[];
}

export interface BetRecommendation {
  id: string;
  game: Game;
  market: string;
  selection: string;
  price_decimal: number | null;
  price_american: number | null;
  predicted_probability: number;
  market_implied_probability: number;
  edge: number;
  expected_value: number;
  kelly_fraction: number;
  recommended_stake_fraction: number;
  confidence_score: number;
  rank: number | null;
  generated_at: string;
  explanation: PredictionExplanation | null;
}

export interface CalibrationBucket {
  bucket_lo: number;
  bucket_hi: number;
  mean_predicted: number;
  actual_win_rate: number;
  num_predictions: number;
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
  calibration_curve: { buckets: CalibrationBucket[] };
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

export interface OddsPoint {
  captured_at: string;
  selection: string;
  price_american: number;
  implied_probability: number | null;
}

export interface Injury {
  id: string;
  player_name: string;
  team_abbreviation: string | null;
  status: string;
  description: string | null;
  report_date: string;
}

export interface ParlayLeg {
  game: Game;
  selection: string;
  price_american: number | null;
  predicted_probability: number;
}

export type ParlayCategory = "best_ev" | "low_variance" | "high_payout";

export interface Parlay {
  id: string;
  num_legs: number;
  category: ParlayCategory;
  combined_probability: number;
  combined_decimal_odds: number;
  combined_ev: number;
  generated_at: string;
  legs: ParlayLeg[];
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
  lineMovement: (gameId: string) => fetchJson<OddsPoint[]>(`/games/${gameId}/line-movement`),
  recentInjuries: (days = 7) => fetchJson<Injury[]>(`/injuries/recent?days=${days}`),
  parlays: (on?: string) => fetchJson<Parlay[]>(`/parlays${on ? `?on=${on}` : ""}`),
};
