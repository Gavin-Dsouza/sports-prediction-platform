"use client";

import { Fragment, useMemo, useState } from "react";

import { LineMovementChart } from "@/components/LineMovementChart";
import { api, type GamePrediction, type OddsPoint } from "@/lib/api";
import { formatGameTime } from "@/lib/format";

function formatProb(p: number | null | undefined): string {
  if (p === null || p === undefined) return "—";
  return `${(p * 100).toFixed(1)}%`;
}

type SortKey = "time" | "probability";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "time", label: "Game time" },
  { value: "probability", label: "Ensemble P(Home Win)" },
];

const DEFAULT_DIRECTION: Record<SortKey, "asc" | "desc"> = {
  time: "asc",
  probability: "desc",
};

function sortValue(prediction: GamePrediction, key: SortKey): number {
  if (key === "probability") {
    return prediction.ensemble_home_win_probability ?? -1;
  }
  return prediction.game.game_datetime ? new Date(prediction.game.game_datetime).getTime() : 0;
}

export function GamesTable({ predictions }: { predictions: GamePrediction[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("time");
  const [direction, setDirection] = useState<"asc" | "desc">(DEFAULT_DIRECTION.time);
  const [expandedGameId, setExpandedGameId] = useState<string | null>(null);
  const [lineMovementCache, setLineMovementCache] = useState<Record<string, OddsPoint[]>>({});
  const [loadingGameId, setLoadingGameId] = useState<string | null>(null);

  function handleSortKeyChange(key: SortKey) {
    setSortKey(key);
    setDirection(DEFAULT_DIRECTION[key]);
  }

  async function toggleExpand(gameId: string) {
    if (expandedGameId === gameId) {
      setExpandedGameId(null);
      return;
    }
    setExpandedGameId(gameId);
    if (!lineMovementCache[gameId]) {
      setLoadingGameId(gameId);
      try {
        const points = await api.lineMovement(gameId);
        setLineMovementCache((cache) => ({ ...cache, [gameId]: points }));
      } catch {
        setLineMovementCache((cache) => ({ ...cache, [gameId]: [] }));
      } finally {
        setLoadingGameId(null);
      }
    }
  }

  const sortedPredictions = useMemo(() => {
    const ascending = [...predictions].sort(
      (a, b) => sortValue(a, sortKey) - sortValue(b, sortKey)
    );
    return direction === "asc" ? ascending : ascending.reverse();
  }, [predictions, sortKey, direction]);

  if (predictions.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No MLB games found for today. Run the daily pipeline (or a historical backfill) first.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2 text-slate-400">
          Sort by
          <select
            value={sortKey}
            onChange={(e) => handleSortKeyChange(e.target.value as SortKey)}
            className="rounded border border-slate-700 bg-panel px-2 py-1 text-slate-100"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setDirection((d) => (d === "asc" ? "desc" : "asc"))}
          className="rounded border border-slate-700 bg-panel px-2 py-1 text-slate-300 hover:text-slate-100"
        >
          {direction === "asc" ? "↑ Ascending" : "↓ Descending"}
        </button>
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="min-w-full divide-y divide-slate-800 text-sm">
          <thead className="bg-panel text-left text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-3">Matchup</th>
              <th className="px-4 py-3">Time</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Score</th>
              <th className="px-4 py-3">Ensemble P(Home Win)</th>
              <th className="px-4 py-3">Per-model</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {sortedPredictions.map(({ game, ensemble_home_win_probability, per_model }) => (
              <Fragment key={game.id}>
                <tr
                  className="cursor-pointer hover:bg-panel/50"
                  onClick={() => toggleExpand(game.id)}
                  title="Click to view line movement"
                >
                  <td className="px-4 py-3 font-medium">
                    {game.away_team.abbreviation} @ {game.home_team.abbreviation}
                  </td>
                  <td className="px-4 py-3 text-slate-400">{formatGameTime(game.game_datetime)}</td>
                  <td className="px-4 py-3 text-slate-400">{game.status}</td>
                  <td className="px-4 py-3 text-slate-400">
                    {game.away_score ?? "-"} : {game.home_score ?? "-"}
                  </td>
                  <td className="px-4 py-3 font-semibold">{formatProb(ensemble_home_win_probability)}</td>
                  <td className="px-4 py-3 text-xs text-slate-400">
                    {Object.entries(per_model)
                      .map(([name, prob]) => `${name}: ${formatProb(prob)}`)
                      .join("  ·  ") || "—"}
                  </td>
                </tr>
                {expandedGameId === game.id && (
                  <tr>
                    <td colSpan={6} className="bg-surface/60 px-4 py-4">
                      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
                        Line movement — home implied win probability
                      </div>
                      {loadingGameId === game.id ? (
                        <p className="text-xs text-slate-500">Loading…</p>
                      ) : (
                        <LineMovementChart points={lineMovementCache[game.id] ?? []} />
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
