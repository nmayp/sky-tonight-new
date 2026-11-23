// A small line chart: thin lines, grey grid, text in the current colour,
// direct labels instead of a legend. Black and blue only; series differ by line style.

export type MiniSeries = { label: string; pts: [number, number][]; blue?: boolean; dash?: "dashed" | "dotted" };
const BLUE = "#1840ff";

export default function MiniChart({ series, xTicks, xFmt = String, yTicks, yFmt = String, yMin, yMax, xLabel, yLabel, marker, aria }: {
  series: MiniSeries[]; xTicks: number[]; xFmt?: (v: number) => string; yTicks: number[]; yFmt?: (v: number) => string;
  yMin: number; yMax: number; xLabel: string; yLabel: string; marker?: number; aria: string;
}) {
  const W = 420, H = 230, L = 44, R = 330 + 40, T = 12, B = 188;
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const px = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (R - L);
  const py = (y: number) => B - ((y - yMin) / (yMax - yMin || 1)) * (B - T);
  // direct labels at the right end, nudged apart so they do not overlap
  const ends = series.map((s, i) => ({ i, y: py(s.pts[s.pts.length - 1][1]) })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 11) ends[k].y = ends[k - 1].y + 11;
  const ly = new Map(ends.map((e) => [e.i, e.y]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={aria}>
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={L} x2={R} y1={py(t)} y2={py(t)} stroke="#efefef" />
          <text x={L - 6} y={py(t) + 4} textAnchor="end" fontSize={10} fill="#8a8a8a">{yFmt(t)}</text>
        </g>
      ))}
      {xTicks.map((t) => <text key={t} x={px(t)} y={B + 15} textAnchor="middle" fontSize={10} fill="#8a8a8a">{xFmt(t)}</text>)}
      <line x1={L} x2={R} y1={B} y2={B} stroke="#cfcfcf" />
      <text x={(L + R) / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="#8a8a8a">{xLabel}</text>
      <text x={10} y={(T + B) / 2} fontSize={10} fill="#8a8a8a" transform={`rotate(-90 10 ${(T + B) / 2})`} textAnchor="middle">{yLabel}</text>
      {marker !== undefined && <line x1={px(marker)} x2={px(marker)} y1={T} y2={B} stroke="currentColor" strokeDasharray="1 3" />}
      {series.map((s, i) => (
        <g key={s.label}>
          <polyline fill="none" stroke={s.blue ? BLUE : "currentColor"} strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round"
            strokeDasharray={s.dash === "dashed" ? "5 3" : s.dash === "dotted" ? "1.5 3" : undefined}
            points={s.pts.map((p) => `${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ")} />
          {s.pts.length <= 12 && s.pts.map((p) => <circle key={p[0]} cx={px(p[0])} cy={py(p[1])} r={2.4} fill={s.blue ? BLUE : "currentColor"} />)}
          <text x={R + 5} y={(ly.get(i) ?? 0) + 3} fontSize={10} fill={s.blue ? BLUE : "currentColor"}>{s.label}</text>
        </g>
      ))}
    </svg>
  );
}
