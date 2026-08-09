"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { api, type GameEmbedding, type NearestGamesResponse } from "@/lib/api";

function formatDate(iso: string): string {
  // `timeZone: "UTC"` is required, not cosmetic: this runs both server-side
  // (SSR, inside the container — UTC) and client-side (hydration, the
  // browser's local timezone) for a "use client" component. Without a fixed
  // zone, toLocaleDateString silently uses the runtime's local zone on each
  // side, which can format the same ISO string as a different calendar day
  // near midnight UTC — server and client then disagree on the rendered
  // text, and React throws a hydration mismatch.
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

function makeGlowTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const texture = new THREE.CanvasTexture(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return texture;

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.4, "rgba(255,255,255,0.4)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  texture.needsUpdate = true;
  return texture;
}

export function ThreeDExplorer({ embeddings }: { embeddings: GameEmbedding[] }) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [neighbors, setNeighbors] = useState<NearestGamesResponse | null>(null);
  const [neighborsLoading, setNeighborsLoading] = useState(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const gamesSorted = useMemo(
    () =>
      [...embeddings].sort(
        (a, b) => new Date(b.game_date).getTime() - new Date(a.game_date).getTime(),
      ),
    [embeddings],
  );

  const filtered = useMemo(() => {
    if (!query.trim()) return gamesSorted.slice(0, 50);
    const q = query.toLowerCase();
    return gamesSorted
      .filter(
        (g) => g.home_team.toLowerCase().includes(q) || g.away_team.toLowerCase().includes(q),
      )
      .slice(0, 50);
  }, [gamesSorted, query]);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setNeighborsLoading(true);
    setNeighborsError(null);
    api
      .nearestGames(selectedId, 10)
      .then((result) => {
        if (!cancelled) setNeighbors(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setNeighborsError(err instanceof Error ? err.message : "Failed to load neighbors");
        }
      })
      .finally(() => {
        if (!cancelled) setNeighborsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId || !containerRef.current || !neighbors) return;
    const container = containerRef.current;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000,
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Normalize raw UMAP coordinates to a fixed, camera-friendly range so
    // orbit distance/zoom behave consistently regardless of the fit's
    // actual output scale.
    const xs = embeddings.map((e) => e.x);
    const ys = embeddings.map((e) => e.y);
    const zs = embeddings.map((e) => e.z);
    const center = new THREE.Vector3(
      (Math.min(...xs) + Math.max(...xs)) / 2,
      (Math.min(...ys) + Math.max(...ys)) / 2,
      (Math.min(...zs) + Math.max(...zs)) / 2,
    );
    const spread =
      Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
        Math.max(...zs) - Math.min(...zs),
      ) || 1;
    const scale = 40 / spread;

    const neighborIds = new Set(neighbors.neighbors.map((n) => n.game_id));
    const NEIGHBOR = new THREE.Color(0x22d3ee);
    const SELECTED = new THREE.Color(0xfacc15);

    const toScaled = (e: GameEmbedding): [number, number, number] => [
      (e.x - center.x) * scale,
      (e.y - center.y) * scale,
      (e.z - center.z) * scale,
    ];

    // Two separate point clouds, not one: with ~9,800 background points
    // additively blended for the glow effect, tightly-packed dim points
    // overlap enough to blow out to white on their own -- a single gold or
    // cyan point sitting in/near a dense cluster would get visually
    // swallowed by that overlap. Rendering the ~11 highlighted points as
    // their own larger, opaque, depth-test-disabled layer on top guarantees
    // they stay visible regardless of how dense the background is nearby.
    const backgroundPositions = new Float32Array(embeddings.length * 3);
    embeddings.forEach((e, i) => {
      const [x, y, z] = toScaled(e);
      backgroundPositions[i * 3] = x;
      backgroundPositions[i * 3 + 1] = y;
      backgroundPositions[i * 3 + 2] = z;
    });

    const backgroundGeometry = new THREE.BufferGeometry();
    backgroundGeometry.setAttribute("position", new THREE.BufferAttribute(backgroundPositions, 3));

    const glowTexture = makeGlowTexture();
    const backgroundMaterial = new THREE.PointsMaterial({
      size: 0.6,
      map: glowTexture,
      color: 0x9ca3af,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const backgroundPoints = new THREE.Points(backgroundGeometry, backgroundMaterial);
    scene.add(backgroundPoints);

    const highlighted = embeddings.filter(
      (e) => e.game_id === selectedId || neighborIds.has(e.game_id),
    );
    const highlightPositions = new Float32Array(highlighted.length * 3);
    const highlightColors = new Float32Array(highlighted.length * 3);
    highlighted.forEach((e, i) => {
      const [x, y, z] = toScaled(e);
      highlightPositions[i * 3] = x;
      highlightPositions[i * 3 + 1] = y;
      highlightPositions[i * 3 + 2] = z;

      const color = e.game_id === selectedId ? SELECTED : NEIGHBOR;
      highlightColors[i * 3] = color.r;
      highlightColors[i * 3 + 1] = color.g;
      highlightColors[i * 3 + 2] = color.b;
    });

    const highlightGeometry = new THREE.BufferGeometry();
    highlightGeometry.setAttribute("position", new THREE.BufferAttribute(highlightPositions, 3));
    highlightGeometry.setAttribute("color", new THREE.BufferAttribute(highlightColors, 3));

    const highlightMaterial = new THREE.PointsMaterial({
      size: 1.6,
      map: glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      depthTest: false,
      sizeAttenuation: true,
    });

    const highlightPoints = new THREE.Points(highlightGeometry, highlightMaterial);
    highlightPoints.renderOrder = 1;
    scene.add(highlightPoints);

    const positions = backgroundPositions;

    // Frame the selected game, not the origin -- its neighborhood is what
    // matters, not the center of the whole (arbitrarily-oriented) projection.
    const selectedIndex = embeddings.findIndex((e) => e.game_id === selectedId);
    const focus =
      selectedIndex >= 0
        ? new THREE.Vector3(
            positions[selectedIndex * 3],
            positions[selectedIndex * 3 + 1],
            positions[selectedIndex * 3 + 2],
          )
        : new THREE.Vector3(0, 0, 0);
    camera.position.set(focus.x + 15, focus.y + 15, focus.z + 15);
    controls.target.copy(focus);
    controls.update();

    let frameId: number;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      controls.dispose();
      backgroundGeometry.dispose();
      backgroundMaterial.dispose();
      highlightGeometry.dispose();
      highlightMaterial.dispose();
      glowTexture.dispose();
      renderer.dispose();
      container.innerHTML = "";
    };
  }, [selectedId, neighbors, embeddings]);

  if (!selectedId) {
    return (
      <div className="rounded-lg border border-slate-800 bg-panel p-6">
        <label className="mb-2 block text-sm font-medium text-slate-300">
          Pick a game to explore
        </label>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by team (e.g. NYY, BOS)..."
          className="mb-4 w-full rounded-md border border-slate-700 bg-surface px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-slate-500 focus:outline-none"
          // Browser extensions (password managers, form fillers) commonly
          // inject attributes like data-has-listeners onto <input> elements
          // before React hydrates, which React otherwise flags as a
          // hydration mismatch even though nothing here is actually wrong.
          suppressHydrationWarning
        />
        <div className="max-h-96 divide-y divide-slate-800 overflow-y-auto">
          {filtered.map((g) => (
            <button
              key={g.game_id}
              onClick={() => setSelectedId(g.game_id)}
              className="flex w-full items-center justify-between px-2 py-2 text-left text-sm hover:bg-slate-800/50"
            >
              <span>
                {g.away_team} @ {g.home_team}
              </span>
              <span className="text-slate-500">{formatDate(g.game_date)}</span>
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="px-2 py-4 text-sm text-slate-500">No games match &quot;{query}&quot;.</div>
          )}
        </div>
      </div>
    );
  }

  const selectedGame = embeddings.find((e) => e.game_id === selectedId);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-sm text-slate-400">
            {selectedGame && (
              <>
                Viewing{" "}
                <span className="text-slate-100">
                  {selectedGame.away_team} @ {selectedGame.home_team}
                </span>{" "}
                · {formatDate(selectedGame.game_date)}
              </>
            )}
          </div>
          <button
            onClick={() => {
              setSelectedId(null);
              setNeighbors(null);
            }}
            className="rounded-md border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
          >
            Pick a different game
          </button>
        </div>
        <div
          ref={containerRef}
          className="h-[600px] w-full overflow-hidden rounded-lg border border-slate-800 bg-black"
        />
        <p className="text-xs text-slate-500">
          Gold = selected game · Cyan = its 10 closest historical games · Drag to orbit, scroll to
          zoom.
        </p>
      </div>

      <div className="rounded-lg border border-slate-800 bg-panel p-4">
        <h2 className="mb-3 text-sm font-medium text-slate-200">Closest historical games</h2>
        {neighborsLoading && <div className="text-sm text-slate-500">Loading…</div>}
        {neighborsError && <div className="text-sm text-negative">{neighborsError}</div>}
        {neighbors && (
          <>
            {neighbors.weighted_home_win_probability !== null && (
              <div className="mb-4 rounded-md border border-slate-700 bg-surface px-3 py-2">
                <div className="text-xs uppercase tracking-wide text-slate-400">
                  Weighted call (closest = most weight)
                </div>
                <div className="mt-1 text-lg font-semibold text-slate-100">
                  {(neighbors.weighted_home_win_probability * 100).toFixed(1)}% home win
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  Based on the {neighbors.neighbors.length} most similar historical games — not
                  the model&apos;s own prediction shown elsewhere on the dashboard.
                </div>
              </div>
            )}
            <ul className="space-y-2 text-sm">
              {neighbors.neighbors.map((n) => (
                <li
                  key={n.game_id}
                  className="flex items-center justify-between rounded-md border border-slate-800 px-2 py-1.5"
                >
                  <div>
                    <div className="text-slate-100">
                      {n.away_team} @ {n.home_team}
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatDate(n.game_date)} ·{" "}
                      {n.home_score !== null && n.away_score !== null
                        ? `${n.away_score}-${n.home_score}${n.home_win ? " (home won)" : " (away won)"}`
                        : "no result"}
                    </div>
                  </div>
                  <div className="text-xs text-slate-400">{(n.similarity * 100).toFixed(2)}%</div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
