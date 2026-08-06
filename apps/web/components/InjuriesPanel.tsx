import type { Injury } from "@/lib/api";

export function InjuriesPanel({ injuries }: { injuries: Injury[] }) {
  if (injuries.length === 0) {
    return <p className="text-sm text-slate-400">No recent injury reports.</p>;
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-panel divide-y divide-slate-800">
      {injuries.map((injury) => (
        <div key={injury.id} className="flex items-center justify-between px-4 py-3 text-sm">
          <div>
            <span className="font-medium text-slate-100">{injury.player_name}</span>
            {injury.team_abbreviation && (
              <span className="ml-2 text-xs text-slate-500">{injury.team_abbreviation}</span>
            )}
            {injury.description && (
              <div className="text-xs text-slate-500">{injury.description}</div>
            )}
          </div>
          <div className="text-right">
            <div className="text-xs font-medium text-negative">{injury.status}</div>
            <div className="text-xs text-slate-500">{injury.report_date}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
