import { BetsTable } from "@/components/BetsTable";
import { GamesTable } from "@/components/GamesTable";
import { InjuriesPanel } from "@/components/InjuriesPanel";
import { ParlaysPanel } from "@/components/ParlaysPanel";
import { StatCard } from "@/components/StatCard";
import { api, type BetRecommendation, type GamePrediction, type Injury, type Parlay } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function TodayPage() {
  let predictions: GamePrediction[] = [];
  let bets: BetRecommendation[] = [];
  let injuries: Injury[] = [];
  let parlays: Parlay[] = [];
  let fetchError: string | null = null;

  try {
    [predictions, bets, injuries, parlays] = await Promise.all([
      api.predictionsToday(),
      api.recommendedBets(),
      api.recentInjuries(),
      api.parlays(),
    ]);
  } catch (error) {
    fetchError =
      error instanceof Error
        ? error.message
        : "Failed to reach the API. Is the backend running?";
  }

  const positiveEvCount = bets.length;
  const bestEdge = bets.length > 0 ? Math.max(...bets.map((b) => b.edge)) : null;

  return (
    <div className="space-y-8">
      {fetchError && (
        <div className="rounded-lg border border-negative/50 bg-negative/10 px-4 py-3 text-sm text-negative">
          {fetchError} — start the stack with <code>docker compose up</code> in{" "}
          <code>infra/</code> and run the daily pipeline task at least once.
        </div>
      )}

      <section>
        <h1 className="mb-4 text-2xl font-semibold">Today&apos;s MLB Slate</h1>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="Games" value={String(predictions.length)} />
          <StatCard label="+EV Bets Found" value={String(positiveEvCount)} tone="positive" />
          <StatCard
            label="Best Edge"
            value={bestEdge !== null ? `+${(bestEdge * 100).toFixed(1)}%` : "—"}
            tone="positive"
          />
          <StatCard
            label="Model"
            value={predictions[0]?.model_version?.slice(0, 8) ?? "—"}
          />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-lg font-medium text-slate-200">Ranked +EV Bets</h2>
        <BetsTable bets={bets} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-medium text-slate-200">Model Probabilities vs. Games</h2>
        <p className="mb-2 text-xs text-slate-500">Click a row to see that game&apos;s line movement.</p>
        <GamesTable predictions={predictions} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-medium text-slate-200">Today&apos;s Parlays</h2>
        <ParlaysPanel parlays={parlays} />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-medium text-slate-200">Recent Injuries</h2>
        <InjuriesPanel injuries={injuries} />
      </section>
    </div>
  );
}
