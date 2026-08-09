"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import {
  api,
  type CompareGamesResponse,
  type GameEmbedding,
  type NearestGamesResponse,
} from "@/lib/api";

// Approximate MLB team primary/accent colors, for the optional "color by
// team" mode. Several teams' actual primary color (navy, black, dark green)
// is nearly invisible against this view's black background, so some entries
// deliberately use that team's brighter secondary/accent color instead of
// their literal brand hex -- this is a visual aid, not an authoritative
// branding reference. A handful of teams also share very similar reds/blues
// by nature of having few available hues; this is an inherent limit of only
// having 30 real team colors to work with, not a bug.
const TEAM_COLORS: Record<string, number> = {
  ARI: 0xa71930,
  ATL: 0xce1141,
  BAL: 0xdf4601,
  BOS: 0xbd3039,
  CHC: 0x0e3386,
  CWS: 0xc4ced4,
  CIN: 0xc6011f,
  CLE: 0xe31937,
  COL: 0x8b4fce,
  DET: 0xfa4616,
  HOU: 0xeb6e1f,
  KC: 0x2f7bd1,
  LAA: 0xba0021,
  LAD: 0x4da3ff,
  MIA: 0x00a3e0,
  MIL: 0xffc52f,
  MIN: 0xd31145,
  NYM: 0xff5910,
  NYY: 0x5b7ba6,
  OAK: 0x2ea86e,
  ATH: 0x2ea86e,
  PHI: 0xe81828,
  PIT: 0xfdb827,
  SD: 0xffc425,
  SEA: 0x0fbfc4,
  SF: 0xfd5a1e,
  STL: 0xc41e3a,
  TB: 0x8fbce6,
  TEX: 0x4f8fd1,
  TOR: 0x3b82c4,
  WSH: 0xab0003,
};
const DEFAULT_TEAM_COLOR = 0x9ca3af;

const SELECTED_COLOR = 0xfacc15; // gold
const NEIGHBOR_COLOR = 0x22d3ee; // cyan
const COMPARE_COLOR = 0xfb923c; // orange

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

/** A soft-edged circular dot texture. `innerStop` controls how much of the
 * radius is solid before fading — a small value (background points) gives a
 * tight, sharp dot with barely any "aura"; a larger value (the highlight
 * layer, which has few points and no crowding problem) gives a softer glow.
 */
function makeDotTexture(innerStop: number): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const texture = new THREE.CanvasTexture(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return texture;

  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(innerStop, "rgba(255,255,255,0.5)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  texture.needsUpdate = true;
  return texture;
}

interface SceneTransform {
  center: THREE.Vector3;
  scale: number;
}

function computeSceneTransform(embeddings: GameEmbedding[]): SceneTransform {
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
  return { center, scale: 40 / spread };
}

function toScaled(e: GameEmbedding, transform: SceneTransform): [number, number, number] {
  return [
    (e.x - transform.center.x) * transform.scale,
    (e.y - transform.center.y) * transform.scale,
    (e.z - transform.center.z) * transform.scale,
  ];
}

export function ThreeDExplorer({ embeddings }: { embeddings: GameEmbedding[] }) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [neighbors, setNeighbors] = useState<NearestGamesResponse | null>(null);
  const [neighborsLoading, setNeighborsLoading] = useState(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);
  const [colorByTeam, setColorByTeam] = useState(false);
  const [locked, setLocked] = useState(false);
  const [compareGameId, setCompareGameId] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<CompareGamesResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);

  // Refs mirroring state that the click handler (attached once per scene
  // rebuild, not on every state change) needs to read without going stale.
  const lockedRef = useRef(locked);
  const selectedIdRef = useRef(selectedId);
  useEffect(() => {
    lockedRef.current = locked;
  }, [locked]);
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  // Refs to persistent three.js objects, shared between the main scene-setup
  // effect and the separate compare-marker effect below -- the marker needs
  // to move without forcing a full scene/point-cloud rebuild on every click.
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const backgroundPointsRef = useRef<THREE.Points | null>(null);
  const highlightPointsRef = useRef<THREE.Points | null>(null);
  const backgroundGameIdsRef = useRef<string[]>([]);
  const highlightGameIdsRef = useRef<string[]>([]);
  const compareMarkerRef = useRef<THREE.Mesh | null>(null);

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

  const embeddingById = useMemo(() => {
    const map = new Map<string, GameEmbedding>();
    for (const e of embeddings) map.set(e.game_id, e);
    return map;
  }, [embeddings]);

  // Neighbor list fetch (unaffected by lock/compare -- always the locked/
  // viewed game's top-10, computed once per game selection).
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

  // Compare fetch: only while locked and a point has been clicked.
  useEffect(() => {
    if (!locked || !compareGameId || !selectedId) {
      setCompareResult(null);
      setCompareError(null);
      return;
    }
    let cancelled = false;
    setCompareLoading(true);
    setCompareError(null);
    api
      .compareGames(selectedId, compareGameId)
      .then((result) => {
        if (!cancelled) setCompareResult(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setCompareError(err instanceof Error ? err.message : "Failed to compare games");
        }
      })
      .finally(() => {
        if (!cancelled) setCompareLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [locked, compareGameId, selectedId]);

  // Clicking a point only makes sense while locked; unlocking clears
  // whatever was being compared so stale state doesn't linger.
  useEffect(() => {
    if (!locked) setCompareGameId(null);
  }, [locked]);

  const handleCanvasClick = useCallback((event: MouseEvent) => {
    if (!lockedRef.current) return;
    const container = containerRef.current;
    const camera = cameraRef.current;
    if (!container || !camera) return;

    const rect = container.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.8 };
    raycaster.setFromCamera(mouse, camera);

    const targets = [backgroundPointsRef.current, highlightPointsRef.current].filter(
      (p): p is THREE.Points => p !== null,
    );
    const intersections = raycaster.intersectObjects(targets, false);
    if (intersections.length === 0) return;

    const hit = intersections[0];
    if (hit.index === undefined) return;
    const gameId =
      hit.object === backgroundPointsRef.current
        ? backgroundGameIdsRef.current[hit.index]
        : highlightGameIdsRef.current[hit.index];
    if (!gameId || gameId === selectedIdRef.current) return;
    setCompareGameId(gameId);
  }, []);

  // Main scene setup: camera, renderer, controls, the two point clouds, the
  // click listener. Rebuilt when the game/neighbors/color mode change --
  // NOT on every compare-click (see the marker effect below for that).
  useEffect(() => {
    if (!selectedId || !containerRef.current || !neighbors) return;
    const container = containerRef.current;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000,
    );
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const transform = computeSceneTransform(embeddings);
    const neighborIds = new Set(neighbors.neighbors.map((n) => n.game_id));

    // Two separate point clouds, not one: with ~9,800 background points,
    // additive blending (needed for a soft glow look) makes densely-packed
    // points overlap and wash out toward solid white on their own. The
    // background layer below uses a sharp, low-opacity, normally-blended
    // dot instead specifically so individual points stay distinguishable
    // even in dense clusters; the highlight layer (a handful of points)
    // keeps the softer glow since it never has a crowding problem, and
    // renders on top (depth-test disabled) so it's never swallowed by the
    // background regardless of local density.
    const backgroundPositions = new Float32Array(embeddings.length * 3);
    const backgroundColors = new Float32Array(embeddings.length * 3);
    backgroundGameIdsRef.current = embeddings.map((e) => e.game_id);
    embeddings.forEach((e, i) => {
      const [x, y, z] = toScaled(e, transform);
      backgroundPositions[i * 3] = x;
      backgroundPositions[i * 3 + 1] = y;
      backgroundPositions[i * 3 + 2] = z;

      const colorHex = colorByTeam
        ? (TEAM_COLORS[e.home_team] ?? DEFAULT_TEAM_COLOR)
        : DEFAULT_TEAM_COLOR;
      const color = new THREE.Color(colorHex);
      backgroundColors[i * 3] = color.r;
      backgroundColors[i * 3 + 1] = color.g;
      backgroundColors[i * 3 + 2] = color.b;
    });

    const backgroundGeometry = new THREE.BufferGeometry();
    backgroundGeometry.setAttribute("position", new THREE.BufferAttribute(backgroundPositions, 3));
    backgroundGeometry.setAttribute("color", new THREE.BufferAttribute(backgroundColors, 3));

    const sharpTexture = makeDotTexture(0.15);
    const backgroundMaterial = new THREE.PointsMaterial({
      size: 0.35,
      map: sharpTexture,
      vertexColors: true,
      transparent: true,
      opacity: colorByTeam ? 0.75 : 0.6,
      depthWrite: false,
      sizeAttenuation: true,
    });

    const backgroundPoints = new THREE.Points(backgroundGeometry, backgroundMaterial);
    backgroundPointsRef.current = backgroundPoints;
    scene.add(backgroundPoints);

    const highlighted = embeddings.filter(
      (e) => e.game_id === selectedId || neighborIds.has(e.game_id),
    );
    highlightGameIdsRef.current = highlighted.map((e) => e.game_id);
    const highlightPositions = new Float32Array(highlighted.length * 3);
    const highlightColors = new Float32Array(highlighted.length * 3);
    highlighted.forEach((e, i) => {
      const [x, y, z] = toScaled(e, transform);
      highlightPositions[i * 3] = x;
      highlightPositions[i * 3 + 1] = y;
      highlightPositions[i * 3 + 2] = z;

      const color = new THREE.Color(e.game_id === selectedId ? SELECTED_COLOR : NEIGHBOR_COLOR);
      highlightColors[i * 3] = color.r;
      highlightColors[i * 3 + 1] = color.g;
      highlightColors[i * 3 + 2] = color.b;
    });

    const highlightGeometry = new THREE.BufferGeometry();
    highlightGeometry.setAttribute("position", new THREE.BufferAttribute(highlightPositions, 3));
    highlightGeometry.setAttribute("color", new THREE.BufferAttribute(highlightColors, 3));

    const glowTexture = makeDotTexture(0.4);
    const highlightMaterial = new THREE.PointsMaterial({
      size: 1.6,
      map: glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const highlightPoints = new THREE.Points(highlightGeometry, highlightMaterial);
    highlightPoints.renderOrder = 1;
    highlightPointsRef.current = highlightPoints;
    scene.add(highlightPoints);

    // Frame the selected game, not the origin -- its neighborhood is what
    // matters, not the center of the whole (arbitrarily-oriented) projection.
    const selectedIndex = embeddings.findIndex((e) => e.game_id === selectedId);
    const focus =
      selectedIndex >= 0
        ? new THREE.Vector3(
            backgroundPositions[selectedIndex * 3],
            backgroundPositions[selectedIndex * 3 + 1],
            backgroundPositions[selectedIndex * 3 + 2],
          )
        : new THREE.Vector3(0, 0, 0);
    camera.position.set(focus.x + 15, focus.y + 15, focus.z + 15);
    controls.target.copy(focus);
    controls.update();

    renderer.domElement.addEventListener("click", handleCanvasClick);

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
      renderer.domElement.removeEventListener("click", handleCanvasClick);
      controls.dispose();
      backgroundGeometry.dispose();
      backgroundMaterial.dispose();
      highlightGeometry.dispose();
      highlightMaterial.dispose();
      sharpTexture.dispose();
      glowTexture.dispose();
      renderer.dispose();
      container.innerHTML = "";
      sceneRef.current = null;
      cameraRef.current = null;
      backgroundPointsRef.current = null;
      highlightPointsRef.current = null;
      compareMarkerRef.current = null;
    };
  }, [selectedId, neighbors, embeddings, colorByTeam, handleCanvasClick]);

  // Compare marker: a small orange sphere at the clicked point, added/moved
  // without touching the rest of the scene (no camera reset, no rebuilding
  // ~9,800 background points on every click).
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (compareMarkerRef.current) {
      scene.remove(compareMarkerRef.current);
      compareMarkerRef.current.geometry.dispose();
      (compareMarkerRef.current.material as THREE.Material).dispose();
      compareMarkerRef.current = null;
    }

    if (!compareGameId) return;
    const point = embeddingById.get(compareGameId);
    if (!point) return;

    const transform = computeSceneTransform(embeddings);
    const [x, y, z] = toScaled(point, transform);

    const geometry = new THREE.SphereGeometry(0.9, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: COMPARE_COLOR, depthTest: false });
    const marker = new THREE.Mesh(geometry, material);
    marker.position.set(x, y, z);
    marker.renderOrder = 2;
    scene.add(marker);
    compareMarkerRef.current = marker;
  }, [compareGameId, embeddings, embeddingById]);

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
            <div className="px-2 py-4 text-sm text-slate-500">
              No games match &quot;{query}&quot;.
            </div>
          )}
        </div>
      </div>
    );
  }

  const selectedGame = embeddingById.get(selectedId);
  const compareGame = compareGameId ? embeddingById.get(compareGameId) : null;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
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
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-xs text-slate-400">
              <input
                type="checkbox"
                checked={colorByTeam}
                onChange={(e) => setColorByTeam(e.target.checked)}
                className="accent-slate-400"
              />
              Color by team
            </label>
            <button
              onClick={() => setLocked((l) => !l)}
              className={`rounded-md border px-3 py-1 text-xs ${
                locked
                  ? "border-orange-400 bg-orange-400/10 text-orange-300"
                  : "border-slate-700 text-slate-300 hover:bg-slate-800"
              }`}
            >
              {locked ? "🔒 Locked — click a dot to compare" : "🔓 Lock & compare"}
            </button>
            <button
              onClick={() => {
                setSelectedId(null);
                setNeighbors(null);
                setLocked(false);
                setCompareGameId(null);
              }}
              className="rounded-md border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
            >
              Pick a different game
            </button>
          </div>
        </div>
        <div
          ref={containerRef}
          className={`h-[600px] w-full overflow-hidden rounded-lg border border-slate-800 bg-black ${
            locked ? "cursor-crosshair" : ""
          }`}
        />
        <p className="text-xs text-slate-500">
          Gold = selected game · Cyan = its 10 closest historical games
          {locked && " · Orange = point you clicked to compare"} · Drag to orbit, scroll to zoom.
        </p>
      </div>

      <div className="space-y-4">
        {locked && (
          <div className="rounded-lg border border-orange-400/40 bg-orange-400/5 p-4">
            <h2 className="mb-2 text-sm font-medium text-orange-300">Comparing</h2>
            {!compareGameId && (
              <p className="text-xs text-slate-400">
                Click any dot in the point cloud to compare it against{" "}
                {selectedGame && (
                  <>
                    {selectedGame.away_team} @ {selectedGame.home_team}
                  </>
                )}
                .
              </p>
            )}
            {compareGameId && compareLoading && (
              <div className="text-sm text-slate-500">Comparing…</div>
            )}
            {compareGameId && compareError && (
              <div className="text-sm text-negative">{compareError}</div>
            )}
            {compareResult && (
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between rounded-md border border-slate-800 bg-surface px-2 py-1.5">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Locked</div>
                    <div className="text-slate-100">
                      {compareResult.game_a.away_team} @ {compareResult.game_a.home_team}
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatDate(compareResult.game_a.game_date)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between rounded-md border border-orange-400/30 bg-surface px-2 py-1.5">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-orange-400">Clicked</div>
                    <div className="text-slate-100">
                      {compareResult.game_b.away_team} @ {compareResult.game_b.home_team}
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatDate(compareResult.game_b.game_date)}
                      {compareResult.game_b.home_score !== null &&
                        compareResult.game_b.away_score !== null &&
                        ` · ${compareResult.game_b.away_score}-${compareResult.game_b.home_score}${
                          compareResult.game_b.home_win ? " (home won)" : " (away won)"
                        }`}
                    </div>
                  </div>
                </div>
                <div className="rounded-md border border-slate-700 px-2 py-1.5 text-center">
                  <span className="text-xs uppercase tracking-wide text-slate-500">
                    Similarity{" "}
                  </span>
                  <span className="text-base font-semibold text-slate-100">
                    {(compareResult.similarity * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}
            {compareGame && (
              <button
                onClick={() => setCompareGameId(null)}
                className="mt-2 text-xs text-slate-500 hover:text-slate-300"
              >
                Clear comparison
              </button>
            )}
          </div>
        )}

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
                    <div className="text-xs text-slate-400">
                      {(n.similarity * 100).toFixed(2)}%
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
