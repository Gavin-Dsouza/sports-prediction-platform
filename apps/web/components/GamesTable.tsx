import type { GamePrediction } from "@/lib/api";

function formatProb(p: number | null | undefined): string {
  if (p === null || p === undefined) return "—";
  return `${(p * 100).toFixed(1)}%`;
}

export function GamesTable({ predictions }: { predictions: GamePrediction[] }) {
  if (predictions.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No MLB games found for today. Run the daily pipeline (or a historical backfill) first.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="min-w-full divide-y divide-slate-800 text-sm">
        <thead className="bg-panel text-left text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="px-4 py-3">Matchup</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Score</th>
            <th className="px-4 py-3">Ensemble P(Home Win)</th>
            <th className="px-4 py-3">Per-model</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {predictions.map(({ game, ensemble_home_win_probability, per_model }) => (
            <tr key={game.id} className="hover:bg-panel/50">
              <td className="px-4 py-3 font-medium">
                {game.away_team.abbreviation} @ {game.home_team.abbreviation}
              </td>
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
          ))}
        </tbody>
      </table>
    </div>
  );
}
