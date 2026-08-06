export function formatGameTime(isoDatetime: string | null): string {
  if (!isoDatetime) return "TBD";
  // Pinned to America/New_York rather than the viewer's local timezone for
  // two reasons: (1) it's the standard convention for displaying US sports
  // game times regardless of who's looking (same as ESPN/sportsbooks), and
  // (2) using the *implicit* local timezone here caused a real Next.js
  // hydration mismatch — the server (inside the Docker container, UTC) and
  // the browser (the viewer's own timezone) each formatted the same instant
  // differently, since `toLocaleString(undefined, ...)` resolves the
  // timezone from whatever environment happens to run it. A fixed timeZone
  // makes server and client compute byte-identical output every time.
  const formatted = new Date(isoDatetime).toLocaleString("en-US", {
    timeZone: "America/New_York",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return `${formatted} ET`;
}

export function formatAmericanOdds(price: number | null): string {
  if (price === null) return "—";
  return price > 0 ? `+${price}` : `${price}`;
}

export function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
