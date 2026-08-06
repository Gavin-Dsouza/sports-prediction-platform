"use client";

import { Fragment, useMemo, useState } from "react";

import { ExplanationPanel } from "@/components/ExplanationPanel";
import type { BetRecommendation } from "@/lib/api";
import { formatAmericanOdds, formatGameTime, pct } from "@/lib/format";

type SortKey = "rank" | "edge" | "confidence" | "ev" | "time";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "rank", label: "Rank (model's ranking)" },
  { value: "edge", label: "Edge" },
  { value: "confidence", label: "Confidence" },
  { value: "ev", label: "EV / $1" },
  { value: "time", label: "Game time" },
];

function sortValue(bet: BetRecommendation, key: SortKey): number {
  switch (key) {
    case "edge":
      return bet.edge;
    case "confidence":
      return bet.confidence_score;
    case "ev":
      return bet.expected_value;
    case "time":
      return bet.game.game_datetime ? new Date(bet.game.game_datetime).getTime() : 0;
    case "rank":
    default:
      // Unranked (null) bets sort last regardless of direction.
      return bet.rank ?? Number.POSITIVE_INFINITY;
  }
}

function teamOnBet(bet: BetRecommendation): string {
  return bet.selection === "home" ? bet.game.home_team.abbreviation : bet.game.away_team.abbreviation;
}

// Each field's "obviously right" default reading: rank 1 first, highest
// edge/confidence/EV first, soonest game first. Switching sort fields resets
// to this default rather than carrying over whatever direction the previous
// field happened to be in.
const DEFAULT_DIRECTION: Record<SortKey, "asc" | "desc"> = {
  rank: "asc",
  edge: "desc",
  confidence: "desc",
  ev: "desc",
  time: "asc",
};

export function BetsTable({ bets }: { bets: BetRecommendation[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [direction, setDirection] = useState<"asc" | "desc">(DEFAULT_DIRECTION.rank);
  const [expandedBetId, setExpandedBetId] = useState<string | null>(null);

  function handleSortKeyChange(key: SortKey) {
    setSortKey(key);
    setDirection(DEFAULT_DIRECTION[key]);
  }

  const sortedBets = useMemo(() => {
    const ascending = [...bets].sort((a, b) => sortValue(a, sortKey) - sortValue(b, sortKey));
    return direction === "asc" ? ascending : ascending.reverse();
  }, [bets, sortKey, direction]);

  if (bets.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No positive-EV bets found for today (or odds haven&apos;t been polled yet).
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
              <th className="px-4 py-3">#</th>
              <th className="px-4 py-3">Matchup</th>
              <th className="px-4 py-3">Time</th>
              <th className="px-4 py-3">Bet On</th>
              <th className="px-4 py-3">Odds</th>
              <th className="px-4 py-3">True P</th>
              <th className="px-4 py-3">Market P</th>
              <th className="px-4 py-3">Edge</th>
              <th className="px-4 py-3">EV / $1</th>
              <th className="px-4 py-3">Kelly Stake</th>
              <th className="px-4 py-3">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {sortedBets.map((bet) => (
              <Fragment key={bet.id}>
                <tr
                  className="cursor-pointer hover:bg-panel/50"
                  onClick={() => setExpandedBetId((id) => (id === bet.id ? null : bet.id))}
                  title="Click to see why the model likes this bet"
                >
                  <td className="px-4 py-3 font-semibold">{bet.rank ?? "—"}</td>
                  <td className="px-4 py-3 font-medium">
                    {bet.game.away_team.abbreviation} @ {bet.game.home_team.abbreviation}
                  </td>
                  <td className="px-4 py-3 text-slate-400">{formatGameTime(bet.game.game_datetime)}</td>
                  <td className="px-4 py-3 font-semibold text-slate-100">
                    {teamOnBet(bet)}
                    <span className="ml-1 text-xs font-normal text-slate-500">ML</span>
                  </td>
                  <td className="px-4 py-3 font-mono">{formatAmericanOdds(bet.price_american)}</td>
                  <td className="px-4 py-3">{pct(bet.predicted_probability)}</td>
                  <td className="px-4 py-3 text-slate-400">{pct(bet.market_implied_probability)}</td>
                  <td className="px-4 py-3 text-positive">+{pct(bet.edge)}</td>
                  <td className="px-4 py-3 text-positive">${bet.expected_value.toFixed(3)}</td>
                  <td className="px-4 py-3">{pct(bet.recommended_stake_fraction)}</td>
                  <td className="px-4 py-3 text-slate-400">{pct(bet.confidence_score)}</td>
                </tr>
                {expandedBetId === bet.id && (
                  <tr>
                    <td colSpan={11} className="bg-surface/60 px-4 py-4">
                      {bet.explanation ? (
                        <ExplanationPanel explanation={bet.explanation} />
                      ) : (
                        <p className="text-xs text-slate-500">
                          No explanation available for this recommendation yet.
                        </p>
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
