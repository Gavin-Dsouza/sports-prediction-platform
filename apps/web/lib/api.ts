// Two different base URLs are needed depending on where the fetch actually
// runs: server-side (Next.js Server Components — the only kind of fetch
// this app had until the 3D view's client-side neighbor lookup) executes
// inside the `web` container's own network namespace, where Docker
// Compose's internal DNS resolves the service name "api"; a client-side
// ("use client") fetch runs in the browser on the host machine, entirely
// outside that Docker network, where only the published host port
// (localhost:8000) is reachable. See infra/docker-compose.yml's `web`
// service for how each is configured.
const SERVER_API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const BROWSER_API_URL = process.env.NEXT_PUBLIC_API_URL_BROWSER ?? "http://localhost:8000";

function apiBaseUrl(): string {
  return typeof window === "undefined" ? SERVER_API_URL : BROWSER_API_URL;
}

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
  game_date: string;
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

export interface GameEmbedding {
  game_id: string;
  x: number;
  y: number;
  z: number;
  game_date: string;
  status: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
}

export interface NeighborGame {
  game_id: string;
  similarity: number;
  game_date: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_win: boolean | null;
}

export interface NearestGamesResponse {
  target_game_id: string;
  neighbors: NeighborGame[];
  weighted_home_win_probability: number | null;
}

export interface GameSummary {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_win: boolean | null;
}

export interface CompareGamesResponse {
  game_a: GameSummary;
  game_b: GameSummary;
  similarity: number;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, { cache: "no-store" });
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
  embeddings: () => fetchJson<GameEmbedding[]>("/embeddings"),
  nearestGames: (gameId: string, k = 10) =>
    fetchJson<NearestGamesResponse>(`/embeddings/${gameId}/neighbors?k=${k}`),
  compareGames: (gameIdA: string, gameIdB: string) =>
    fetchJson<CompareGamesResponse>(`/embeddings/${gameIdA}/compare/${gameIdB}`),
};
