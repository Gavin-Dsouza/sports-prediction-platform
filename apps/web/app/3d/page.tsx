import { ThreeDExplorer } from "@/components/ThreeDExplorer";
import { api, type GameEmbedding } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ThreeDPage() {
  let embeddings: GameEmbedding[] = [];
  let fetchError: string | null = null;

  try {
    embeddings = await api.embeddings();
  } catch (error) {
    fetchError = error instanceof Error ? error.message : "Failed to reach the API.";
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">3D Game Explorer</h1>
        <p className="mt-1 text-sm text-slate-400">
          Every game plotted by feature similarity — pick a game to see it, and the historical
          games most like it, in the point cloud.
        </p>
      </div>

      {fetchError && (
        <div className="rounded-lg border border-negative/50 bg-negative/10 px-4 py-3 text-sm text-negative">
          {fetchError} — start the stack with <code>docker compose up</code> in{" "}
          <code>infra/</code> and run the embeddings script at least once.
        </div>
      )}

      {!fetchError && embeddings.length === 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-4 py-3 text-sm text-slate-400">
          No embeddings computed yet — run the embeddings script to populate the 3D view.
        </div>
      )}

      {embeddings.length > 0 && <ThreeDExplorer embeddings={embeddings} />}
    </div>
  );
}
