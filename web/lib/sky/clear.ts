// Chance of a clear sky for any point in Spain and any date: the model's forecast when a forecast exists,
// fading to the model's own climatology as the lead time grows. The weights come from skill.json (measured).

export type Grid = { lat: number[]; lon: number[]; alt: number[]; clim: (number | null)[][] };
export type Forecast = { cycle: string; valid: { utc: string; lead_h: number }[]; p: (number | null)[][] };
export type Lead = { lead_h: number; n: number; model: number; raw: number; clim: number; best_weight: number; blend: number; auc: number };
export type Lights = Record<"mainland" | "canary", { lon0: number; lat1: number; step: number; w: number; h: number; data: number[] }>;

const MAIN = { lat0: 35.5, lon0: -9.5, nLat: 18, nLon: 29 };
const CAN = { lat0: 27.5, lon0: -18.5, nLat: 5, nLon: 11 };

export class ClearSky {
  fc: Forecast;
  validMs: number[];
  cycleMs: number;
  weights: [number, number][];
  constructor(public grid: Grid, fc0: Forecast, leads: Lead[], public lights: Lights) {
    const order = fc0.valid.map((v, i) => i).sort((a, b) => Date.parse(fc0.valid[a].utc) - Date.parse(fc0.valid[b].utc));
    this.fc = { cycle: fc0.cycle, valid: order.map((i) => fc0.valid[i]), p: order.map((i) => fc0.p[i]) };
    this.validMs = this.fc.valid.map((v) => Date.parse(v.utc));
    this.cycleMs = Date.parse(this.fc.cycle);
    // forecast weight by lead, forced to be non-increasing and to start at the lead-9 value
    let w = 1;
    this.weights = leads.slice().sort((a, b) => a.lead_h - b.lead_h).map((l) => { w = Math.min(w, l.best_weight); return [l.lead_h, w] as [number, number]; });
  }

  private cell(lat: number, lon: number) {
    const blk = lat < 31 && lon < -12 ? CAN : MAIN;
    const fy = (lat - blk.lat0) / 0.5, fx = (lon - blk.lon0) / 0.5;
    const y0 = Math.max(0, Math.min(blk.nLat - 2, Math.floor(fy))), x0 = Math.max(0, Math.min(blk.nLon - 2, Math.floor(fx)));
    const ty = Math.max(0, Math.min(1, fy - y0)), tx = Math.max(0, Math.min(1, fx - x0));
    const base = blk === CAN ? MAIN.nLat * MAIN.nLon : 0;
    const idx = (y: number, x: number) => base + y * blk.nLon + x;
    return { i: [idx(y0, x0), idx(y0, x0 + 1), idx(y0 + 1, x0), idx(y0 + 1, x0 + 1)], w: [(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx] };
  }

  private interp(vals: (number | null)[], c: { i: number[]; w: number[] }) {
    let s = 0, ws = 0;
    c.i.forEach((k, n) => { const v = vals[k]; if (v != null) { s += v * c.w[n]; ws += c.w[n]; } });
    return ws > 0 ? s / ws : null;
  }

  /** Climatological chance (0..1) of a clear sky on a night of this date, smoothed between month centres. */
  clim(lat: number, lon: number, d: Date) {
    return this.climCell(this.cell(lat, lon), d);
  }

  climCell(c: { i: number[]; w: number[] }, d: Date) {
    const m = d.getUTCMonth(), day = d.getUTCDate();
    const f = (day - 15) / 30;
    const m2 = f >= 0 ? (m + 1) % 12 : (m + 11) % 12;
    const a = this.interp(this.grid.clim.map((r) => r[m]), c), b = this.interp(this.grid.clim.map((r) => r[m2]), c);
    if (a == null) return 0.5;
    return ((b == null ? a : a * (1 - Math.abs(f)) + b * Math.abs(f))) / 100;
  }

  weight(leadH: number) {
    const w = this.weights;
    if (leadH <= w[0][0]) return w[0][1];
    for (let i = 1; i < w.length; i++) if (leadH <= w[i][0]) { const t = (leadH - w[i - 1][0]) / (w[i][0] - w[i - 1][0]); return w[i - 1][1] * (1 - t) + w[i][1] * t; }
    return w[w.length - 1][1];
  }

  /** Model forecast (0..1) at an instant from the stored GFS run, or null if outside the stored period. */
  forecast(lat: number, lon: number, t: number) {
    return this.forecastCell(this.cell(lat, lon), t);
  }

  forecastCell(c: { i: number[]; w: number[] }, t: number) {
    const v = this.validMs;
    if (t < v[0] || t > v[v.length - 1]) return null;
    let k = v.findIndex((x) => x >= t);
    if (k <= 0) k = 1;
    if (v[k] - v[k - 1] > 4 * 3600e3 && Math.min(t - v[k - 1], v[k] - t) > 1.6 * 3600e3) return null; // gap between nights
    const a = this.interp(this.fc.p[k - 1], c), b = this.interp(this.fc.p[k], c);
    if (a == null || b == null) return null;
    const u = Math.max(0, Math.min(1, (t - v[k - 1]) / (v[k] - v[k - 1])));
    return (a * (1 - u) + b * u) / 100;
  }

  chance(lat: number, lon: number, when: Date) {
    return this.chanceCell(this.cell(lat, lon), when);
  }

  chanceIdx(i: number, when: Date) {
    return this.chanceCell({ i: [i], w: [1] }, when);
  }

  chanceCell(c: { i: number[]; w: number[] }, when: Date) {
    const t = when.getTime();
    const lead = (t - this.cycleMs) / 3600e3;
    const cl = this.climCell(c, when);
    const pf = lead >= 3 ? this.forecastCell(c, t) : null;
    if (pf == null) return { p: cl, pf: null as number | null, clim: cl, w: 0, lead };
    const w = this.weight(lead);
    return { p: w * pf + (1 - w) * cl, pf, clim: cl, w, lead };
  }

  /** Darkness index: mean satellite night light within 25 km (0-255) and its class. */
  darkness(lat: number, lon: number) {
    const L = lat < 31 && lon < -12 ? this.lights.canary : this.lights.mainland;
    const x = Math.round((lon - L.lon0) / L.step), y = Math.round((L.lat1 - lat) / L.step);
    const v = L.data[Math.max(0, Math.min(L.h - 1, y)) * L.w + Math.max(0, Math.min(L.w - 1, x))];
    const cls = v <= 20 ? "very dark" : v <= 40 ? "dark" : v <= 70 ? "moderate" : v <= 120 ? "bright" : "very bright";
    return { v, cls };
  }
}

export const pct = (p: number) => `${Math.round(p * 100)} %`;
