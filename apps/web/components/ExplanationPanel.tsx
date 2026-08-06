import type { PredictionExplanation } from "@/lib/api";

function featureLabel(name: string): string {
  return name.replace(/_/g, " ");
}

export function ExplanationPanel({ explanation }: { explanation: PredictionExplanation }) {
  const maxAbsContribution = Math.max(
    0.001,
    ...explanation.top_reasons.map((r) => Math.abs(r.contribution ?? 0))
  );

  return (
    <div className="space-y-4 text-sm">
      {explanation.top_reasons.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            Why the ensemble likes this (XGBoost + Logistic Regression, SHAP-weighted{" "}
            {(explanation.shap_model_weight * 100).toFixed(0)}% of blend)
          </div>
          <div className="space-y-1.5">
            {explanation.top_reasons.map((reason) => {
              const contribution = reason.contribution ?? 0;
              const widthPct = (Math.abs(contribution) / maxAbsContribution) * 100;
              const positive = contribution >= 0;
              return (
                <div key={reason.feature} className="flex items-center gap-2">
                  <div className="w-40 shrink-0 truncate text-xs text-slate-400" title={reason.feature}>
                    {featureLabel(reason.feature)}
                  </div>
                  <div className="h-3 flex-1 rounded bg-slate-800">
                    <div
                      className={`h-3 rounded ${positive ? "bg-positive" : "bg-negative"}`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <div className="w-16 shrink-0 text-right text-xs text-slate-400">
                    {contribution >= 0 ? "+" : ""}
                    {contribution.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {explanation.also_considered.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
            Also factored in (Elo / Poisson)
          </div>
          <div className="text-xs text-slate-400">
            {explanation.also_considered.map((r) => featureLabel(r.feature)).join("  ·  ")}
          </div>
        </div>
      )}

      {explanation.similar_games.length > 0 && (
        <div>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">
            Similar historical games
          </div>
          <div className="space-y-1">
            {explanation.similar_games.map((g) => (
              <div key={g.game_id} className="flex justify-between text-xs text-slate-400">
                <span>
                  {g.away_team} @ {g.home_team} ({g.away_score ?? "-"}:{g.home_score ?? "-"})
                </span>
                <span>{(g.similarity * 100).toFixed(1)}% similar</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
