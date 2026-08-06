import type { Parlay, ParlayCategory } from "@/lib/api";
import { formatAmericanOdds, pct } from "@/lib/format";

const CATEGORY_LABEL: Record<ParlayCategory, string> = {
  best_ev: "Best EV",
  low_variance: "Low Variance",
  high_payout: "High Payout",
};

const CATEGORY_ORDER: ParlayCategory[] = ["best_ev", "low_variance", "high_payout"];

function legSelectionLabel(leg: Parlay["legs"][number]): string {
  const team = leg.selection === "home" ? leg.game.home_team.abbreviation : leg.game.away_team.abbreviation;
  return `${team} ML`;
}

export function ParlaysPanel({ parlays }: { parlays: Parlay[] }) {
  if (parlays.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No parlays generated for today yet (needs at least two +EV bets on different games).
      </p>
    );
  }

  const legCounts = [...new Set(parlays.map((p) => p.num_legs))].sort((a, b) => a - b);

  return (
    <div className="space-y-6">
      {legCounts.map((legCount) => {
        const byCategory = new Map(parlays.filter((p) => p.num_legs === legCount).map((p) => [p.category, p]));

        return (
          <div key={legCount}>
            <h3 className="mb-2 text-sm font-medium text-slate-300">{legCount}-Leg Parlays</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {CATEGORY_ORDER.map((category) => {
                const parlay = byCategory.get(category);
                if (!parlay) return null;

                return (
                  <div key={category} className="rounded-lg border border-slate-800 bg-panel p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                        {CATEGORY_LABEL[category]}
                      </span>
                      <span className="text-xs text-positive">${parlay.combined_ev.toFixed(2)} EV</span>
                    </div>
                    <div className="space-y-1 divide-y divide-slate-800/60">
                      {parlay.legs.map((leg, i) => (
                        <div key={i} className="flex items-center justify-between py-1.5 text-sm">
                          <span className="text-slate-100">{legSelectionLabel(leg)}</span>
                          <span className="font-mono text-slate-400">{formatAmericanOdds(leg.price_american)}</span>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-2 text-xs text-slate-500">
                      <span>{pct(parlay.combined_probability)} win prob</span>
                      <span>{parlay.combined_decimal_odds.toFixed(2)}x payout</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
