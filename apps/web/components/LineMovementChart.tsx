"use client";

import { useState } from "react";

import type { OddsPoint } from "@/lib/api";

// Fixed timezone for the same reason as lib/format.ts::formatGameTime — keeps
// this chart's own axis/tooltip times consistent with the rest of the app
// and avoids the timezone-dependent-formatting hydration pitfall entirely.
function formatEasternShort(ms: number): string {
  return new Date(ms).toLocaleString("en-US", {
    timeZone: "America/New_York",
    month: "short",
    day: "numeric",
    hour: "numeric",
  });
}

const WIDTH = 480;
const HEIGHT = 220;
const PAD = { top: 16, right: 16, bottom: 32, left: 44 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;

const SERIES_COLOR = "#3987e5"; // same validated accent as CalibrationChart

export function LineMovementChart({ points }: { points: OddsPoint[] }) {
  const [hovered, setHovered] = useState<number | null>(null);

  const homePoints = points
    .filter((p) => p.selection === "home" && p.implied_probability !== null)
    .sort((a, b) => new Date(a.captured_at).getTime() - new Date(b.captured_at).getTime());

  if (homePoints.length === 0) {
    return <p className="text-xs text-slate-500">No odds history yet for this game.</p>;
  }

  const times = homePoints.map((p) => new Date(p.captured_at).getTime());
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const timeSpan = maxTime - minTime || 1;

  function x(t: number): number {
    return PAD.left + ((t - minTime) / timeSpan) * PLOT_W;
  }
  function y(prob: number): number {
    return PAD.top + (1 - prob) * PLOT_H;
  }

  const linePath = homePoints
    .map((p, i) => {
      const px = x(new Date(p.captured_at).getTime());
      const py = y(p.implied_probability ?? 0);
      return `${i === 0 ? "M" : "L"} ${px} ${py}`;
    })
    .join(" ");

  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full max-w-lg"
        role="img"
        aria-label="Home team implied win probability over time"
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} y1={y(t)} x2={PAD.left + PLOT_W} y2={y(t)} stroke="#1e293b" strokeWidth={1} />
            <text x={PAD.left - 8} y={y(t) + 3} fill="#64748b" fontSize={10} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}

        <path d={linePath} fill="none" stroke={SERIES_COLOR} strokeWidth={2} strokeLinecap="round" />

        {homePoints.map((p, i) => (
          <circle
            key={i}
            cx={x(new Date(p.captured_at).getTime())}
            cy={y(p.implied_probability ?? 0)}
            r={hovered === i ? 5 : 4}
            fill={SERIES_COLOR}
            stroke="#0f172a"
            strokeWidth={1.5}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
            className="cursor-pointer"
          />
        ))}

        <text x={PAD.left} y={HEIGHT - 4} fill="#64748b" fontSize={10} textAnchor="start">
          {formatEasternShort(minTime)}
        </text>
        <text x={PAD.left + PLOT_W} y={HEIGHT - 4} fill="#64748b" fontSize={10} textAnchor="end">
          {formatEasternShort(maxTime)} ET
        </text>
      </svg>

      {hovered !== null && (
        <div
          className="pointer-events-none absolute rounded border border-slate-700 bg-panel px-3 py-2 text-xs shadow-lg"
          style={{
            left: `${(x(new Date(homePoints[hovered].captured_at).getTime()) / WIDTH) * 100}%`,
            top: `${(y(homePoints[hovered].implied_probability ?? 0) / HEIGHT) * 100}%`,
            transform: "translate(-50%, -120%)",
          }}
        >
          <div className="font-semibold text-slate-100">
            {((homePoints[hovered].implied_probability ?? 0) * 100).toFixed(1)}% home
          </div>
          <div className="text-slate-400">{homePoints[hovered].price_american > 0 ? "+" : ""}{homePoints[hovered].price_american}</div>
          <div className="text-slate-500">
            {formatEasternShort(new Date(homePoints[hovered].captured_at).getTime())} ET
          </div>
        </div>
      )}
    </div>
  );
}
