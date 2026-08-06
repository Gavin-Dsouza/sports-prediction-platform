"use client";

import { useState } from "react";

import type { CalibrationBucket } from "@/lib/api";

const WIDTH = 480;
const HEIGHT = 320;
const PAD = { top: 16, right: 16, bottom: 40, left: 48 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;

// Validated dark-mode series color from the platform's data-viz palette
// (skill: dataviz — references/palette.md, slot 1 "blue"). Single series, so
// no legend is required, but the color is still the deliberate accent, not
// an arbitrary pick.
const SERIES_COLOR = "#3987e5";

function x(value: number): number {
  return PAD.left + value * PLOT_W;
}

function y(value: number): number {
  return PAD.top + (1 - value) * PLOT_H;
}

const TICKS = [0, 0.25, 0.5, 0.75, 1];

export function CalibrationChart({ buckets }: { buckets: CalibrationBucket[] }) {
  const [hovered, setHovered] = useState<number | null>(null);

  if (buckets.length === 0) {
    return <p className="text-sm text-slate-400">No calibration data for this backtest.</p>;
  }

  const sorted = [...buckets].sort((a, b) => a.mean_predicted - b.mean_predicted);
  const linePath = sorted
    .map((b, i) => `${i === 0 ? "M" : "L"} ${x(b.mean_predicted)} ${y(b.actual_win_rate)}`)
    .join(" ");

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full max-w-lg" role="img" aria-label="Calibration chart: predicted probability vs actual win rate">
        {/* Recessive gridlines */}
        {TICKS.map((t) => (
          <g key={t}>
            <line x1={x(t)} y1={PAD.top} x2={x(t)} y2={PAD.top + PLOT_H} stroke="#1e293b" strokeWidth={1} />
            <line x1={PAD.left} y1={y(t)} x2={PAD.left + PLOT_W} y2={y(t)} stroke="#1e293b" strokeWidth={1} />
            <text x={x(t)} y={PAD.top + PLOT_H + 16} fill="#64748b" fontSize={10} textAnchor="middle">
              {Math.round(t * 100)}%
            </text>
            <text x={PAD.left - 8} y={y(t) + 3} fill="#64748b" fontSize={10} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}

        {/* Perfect-calibration reference diagonal — a guide, not a data series */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="#475569"
          strokeWidth={1.5}
          strokeDasharray="4 4"
        />

        {/* Actual calibration line */}
        <path d={linePath} fill="none" stroke={SERIES_COLOR} strokeWidth={2} strokeLinecap="round" />

        {sorted.map((b, i) => (
          <circle
            key={i}
            cx={x(b.mean_predicted)}
            cy={y(b.actual_win_rate)}
            r={hovered === i ? 6 : 5}
            fill={SERIES_COLOR}
            stroke="#0f172a"
            strokeWidth={1.5}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered((h) => (h === i ? null : h))}
            className="cursor-pointer"
          />
        ))}

        <text
          x={PAD.left + PLOT_W / 2}
          y={HEIGHT - 4}
          fill="#94a3b8"
          fontSize={11}
          textAnchor="middle"
        >
          Predicted probability
        </text>
        <text
          x={-(PAD.top + PLOT_H / 2)}
          y={14}
          fill="#94a3b8"
          fontSize={11}
          textAnchor="middle"
          transform="rotate(-90)"
        >
          Actual win rate
        </text>
      </svg>

      {hovered !== null && (
        <div
          className="pointer-events-none absolute rounded border border-slate-700 bg-panel px-3 py-2 text-xs shadow-lg"
          style={{
            left: `${(x(sorted[hovered].mean_predicted) / WIDTH) * 100}%`,
            top: `${(y(sorted[hovered].actual_win_rate) / HEIGHT) * 100}%`,
            transform: "translate(-50%, -120%)",
          }}
        >
          <div className="font-semibold text-slate-100">
            {Math.round(sorted[hovered].bucket_lo * 100)}%-{Math.round(sorted[hovered].bucket_hi * 100)}%
            bucket
          </div>
          <div className="text-slate-400">
            Predicted: {(sorted[hovered].mean_predicted * 100).toFixed(1)}%
          </div>
          <div className="text-slate-400">
            Actual: {(sorted[hovered].actual_win_rate * 100).toFixed(1)}%
          </div>
          <div className="text-slate-500">n = {sorted[hovered].num_predictions}</div>
        </div>
      )}
      <p className="mt-1 text-xs text-slate-500">
        Closer to the dashed diagonal = better calibrated (predicted probability matches
        reality).
      </p>
    </div>
  );
}
