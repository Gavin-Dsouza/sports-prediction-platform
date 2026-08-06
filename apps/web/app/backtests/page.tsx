import { CalibrationChart } from "@/components/CalibrationChart";
import { StatCard } from "@/components/StatCard";
import { api, type BacktestRun } from "@/lib/api";

export const dynamic = "force-dynamic";

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export default async function BacktestsPage() {
  let runs: BacktestRun[] = [];
  let fetchError: string | null = null;

  try {
    runs = await api.backtests();
  } catch (error) {
    fetchError = error instanceof Error ? error.message : "Failed to reach the API.";
  }

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-semibold">Backtest Results</h1>

      {fetchError && (
        <div className="rounded-lg border border-negative/50 bg-negative/10 px-4 py-3 text-sm text-negative">
          {fetchError}
        </div>
      )}

      {runs.length === 0 && !fetchError && (
        <p className="text-sm text-slate-400">
          No backtest runs yet. Trigger one via <code>packages.evaluation.backtest.run_walk_forward_backtest</code>{" "}
          (see the README for the CLI invocation).
        </p>
      )}

      <div className="space-y-6">
        {runs.map((run) => (
          <div key={run.id} className="rounded-lg border border-slate-800 bg-panel p-4">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="font-medium">
                  {run.sport} · {run.predictor_name} · {run.start_date} → {run.end_date}
                </div>
                <div className="text-xs text-slate-500">
                  model_version={run.model_version} · run at {new Date(run.created_at).toLocaleString()}
                </div>
              </div>
            </div>
            {run.results.map((result) => (
              <div key={result.market} className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
                <StatCard label="Market" value={result.market} />
                <StatCard label="Bets" value={String(result.num_bets)} />
                <StatCard label="Accuracy" value={pct(result.accuracy)} />
                <StatCard label="Log Loss" value={result.log_loss?.toFixed(3) ?? "—"} />
                <StatCard label="Brier" value={result.brier_score?.toFixed(3) ?? "—"} />
                <StatCard
                  label="ROI"
                  value={pct(result.roi)}
                  tone={result.roi && result.roi > 0 ? "positive" : "negative"}
                />
                <StatCard label="Max Drawdown" value={result.max_drawdown?.toFixed(2) ?? "—"} />
                <StatCard label="Max Losing Streak" value={String(result.max_losing_streak ?? "—")} />
              </div>
            ))}
            {run.results.map((result) => (
              <div key={`${result.market}-calibration`} className="mt-4">
                <h3 className="mb-2 text-sm font-medium text-slate-300">
                  Calibration — {result.market}
                </h3>
                <CalibrationChart buckets={result.calibration_curve.buckets} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
