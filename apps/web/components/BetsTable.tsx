import type { BetRecommendation } from "@/lib/api";

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function BetsTable({ bets }: { bets: BetRecommendation[] }) {
  if (bets.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No positive-EV bets found for today (or odds haven&apos;t been polled yet).
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="min-w-full divide-y divide-slate-800 text-sm">
        <thead className="bg-panel text-left text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Market</th>
            <th className="px-4 py-3">Selection</th>
            <th className="px-4 py-3">True P</th>
            <th className="px-4 py-3">Market P</th>
            <th className="px-4 py-3">Edge</th>
            <th className="px-4 py-3">EV / $1</th>
            <th className="px-4 py-3">Kelly Stake</th>
            <th className="px-4 py-3">Confidence</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {bets.map((bet) => (
            <tr key={bet.id} className="hover:bg-panel/50">
              <td className="px-4 py-3 font-semibold">{bet.rank ?? "—"}</td>
              <td className="px-4 py-3 text-slate-400">{bet.market}</td>
              <td className="px-4 py-3">{bet.selection}</td>
              <td className="px-4 py-3">{pct(bet.predicted_probability)}</td>
              <td className="px-4 py-3 text-slate-400">{pct(bet.market_implied_probability)}</td>
              <td className="px-4 py-3 text-positive">+{pct(bet.edge)}</td>
              <td className="px-4 py-3 text-positive">${bet.expected_value.toFixed(3)}</td>
              <td className="px-4 py-3">{pct(bet.recommended_stake_fraction)}</td>
              <td className="px-4 py-3 text-slate-400">{pct(bet.confidence_score)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
