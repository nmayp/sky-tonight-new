// What you will actually see: turns positions into plain verdicts using how dark the place is, the Moon and twilight.
import * as A from "astronomy-engine";
import * as sat from "satellite.js";
import { SHOWERS, type Report } from "./astro";

export type Tle = Record<string, { l1: string; l2: string; epoch: string }>;
export type Seen = { group: string; name: string; where: string; how: string; verdict: "Easy" | "Possible" | "Hard"; why: string; rank: number };

const BASE: Record<string, number> = { "very dark": 6.4, dark: 5.9, moderate: 4.9, bright: 3.9, "very bright": 3.0 };
export const DARK_FACTOR: Record<string, number> = { "very dark": 1, dark: 0.85, moderate: 0.6, bright: 0.35, "very bright": 0.2 };
const dir = (az: number) => ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][Math.round(az / 45) % 8];

/** Faintest star a person can see here and now (magnitude), a rule of thumb from the light-pollution class, Moon and twilight. */
export function limitingMag(cls: string, r: Report) {
  let lm = BASE[cls] ?? 5;
  if (r.moon.alt > 0) lm -= 2.2 * r.moon.illum * Math.max(0.3, Math.min(1, r.moon.alt / 30));
  if (r.sunAlt > -18) lm -= Math.max(0, Math.min(6, ((r.sunAlt + 18) / 12) * 6));
  return lm;
}

const EXT = new Set(["g", "pn", "en", "rn", "snr", "dn", "gal", "neb"]);

export function whatYouSee(r: Report, cls: string, passes: Pass[]): { lm: number; rows: Seen[]; hidden: number } {
  const lm = limitingMag(cls, r);
  const rows: Seen[] = [];
  const dark = r.sunAlt < -6;
  let hidden = 0;
  if (r.moon.alt > 0) rows.push({ group: "Moon", name: `${r.moon.phase} Moon`, where: `${r.moon.alt.toFixed(0)}° up, ${dir(r.moon.az)}`, how: "naked eye", verdict: "Easy", why: `${Math.round(r.moon.illum * 100)} % lit`, rank: 0 });
  for (const p of r.planets) {
    const m = p.mag ?? 9;
    if (p.alt < 8) { hidden++; continue; }
    if (r.sunAlt > -3 && m > -3) { hidden++; continue; }
    const how = m <= lm - 0.3 ? "naked eye" : m <= lm + 3.5 ? "binoculars" : "telescope";
    const verdict = m <= lm - 2 ? "Easy" : m <= lm - 0.3 ? "Possible" : "Hard";
    rows.push({ group: "Planet", name: p.name, where: `${p.alt.toFixed(0)}° up, ${dir(p.az)}`, how, verdict, why: `magnitude ${m.toFixed(1)}`, rank: 1 + (m + 5) / 20 });
  }
  for (const s of r.showers) {
    const sh = SHOWERS.find((x) => x.name === s.name)!;
    if (s.radiantAlt < 15 || !dark) { hidden++; continue; }
    const rate = Math.round(s.zhr * Math.sin((s.radiantAlt * Math.PI) / 180) * Math.pow(2.5, lm - 6.5) * Math.exp(-Math.abs(s.daysFromPeak) / (sh.halfWindow * 0.8)));
    rows.push({ group: "Meteors", name: `${s.name} meteor shower`, where: `radiant ${s.radiantAlt.toFixed(0)}° up`, how: "naked eye, lie back and look wide", verdict: rate >= 10 ? "Easy" : rate >= 3 ? "Possible" : "Hard", why: `about ${Math.max(rate, 0)} an hour here (peak ${s.peak})`, rank: 2 });
  }
  for (const pa of passes) rows.push({ group: "Satellite", name: pa.name, where: `max ${pa.maxEl.toFixed(0)}° up, ${dir(pa.maxAz)}`, how: "naked eye, moving dot", verdict: pa.maxEl >= 40 ? "Easy" : "Possible", why: `${pa.start} to ${pa.end} local`, rank: 2.5 });
  const mw = r.milkyWay;
  if (dark && mw.coreAlt > 10) rows.push({ group: "Milky Way", name: "Milky Way core", where: `${mw.coreAlt.toFixed(0)}° up, ${dir(mw.coreAz)}`, how: "naked eye", verdict: lm >= 5.6 ? "Easy" : lm >= 4.8 ? "Possible" : "Hard", why: lm >= 5.6 ? "dark sky" : "needs a darker sky", rank: 3 });
  const stars = r.stars.filter((s) => (s.mag ?? 9) <= lm && dark).slice(0, 6);
  for (const s of stars) rows.push({ group: "Star", name: `${s.name} (${s.sub})`, where: `${s.alt.toFixed(0)}° up, ${dir(s.az)}`, how: "naked eye", verdict: "Easy", why: `magnitude ${(s.mag ?? 0).toFixed(1)}`, rank: 4 + (s.mag ?? 0) / 10 });
  let deepN = 0;
  for (const d of r.deep) {
    if (!dark) break;
    const eff = (d.mag ?? 9) + (EXT.has(d.kind) ? 1.4 : 0.4);
    const near = parseFloat(d.sub ?? "99") < 20 && r.moon.alt > 0 && r.moon.illum > 0.4;
    const how = eff <= lm ? "naked eye" : eff <= lm + 3.3 ? "binoculars or small telescope" : "telescope";
    if (how === "telescope" && deepN >= 4) continue;
    if (deepN >= 9) break;
    deepN++;
    rows.push({ group: "Deep sky", name: d.name, where: `${d.alt.toFixed(0)}° up, ${dir(d.az)}`, how, verdict: near ? "Hard" : eff <= lm ? "Possible" : eff <= lm + 3.3 ? "Possible" : "Hard", why: near ? "too close to the Moon" : `magnitude ${(d.mag ?? 0).toFixed(1)}`, rank: 5 + (d.mag ?? 0) / 10 });
  }
  return { lm, rows: rows.sort((a, b) => a.rank - b.rank), hidden };
}

export type Pass = { name: string; startMs: number; start: string; end: string; maxEl: number; maxAz: number };

/** Passes of bright satellites that a person can see: above 10 degrees, sunlit, with the observer in darkness. */
export function satellitePasses(tle: Tle, lat: number, lon: number, elev: number, fromMs: number, toMs: number, fmt: (d: Date) => string): { passes: Pass[]; staleDays: number } {
  const out: Pass[] = [];
  const obs = { latitude: (lat * Math.PI) / 180, longitude: (lon * Math.PI) / 180, height: elev / 1000 };
  const o = new A.Observer(lat, lon, elev);
  let stale = 0;
  for (const [name, t] of Object.entries(tle)) {
    const rec = sat.twoline2satrec(t.l1, t.l2);
    stale = Math.max(stale, (toMs - Date.parse(t.epoch)) / 86400e3);
    let cur: { s: number; e: number; el: number; az: number } | null = null;
    let sunAltCache = { t: -1, v: 0 };
    for (let ms = fromMs; ms <= toMs; ms += 15000) {
      const d = new Date(ms);
      if (ms - sunAltCache.t > 300000) {
        const e = A.Equator(A.Body.Sun, d, o, true, true);
        sunAltCache = { t: ms, v: A.Horizon(d, o, e.ra, e.dec, "normal").altitude };
      }
      let visible = false, el = 0, az = 0;
      if (sunAltCache.v < -6) {
        const pv = sat.propagate(rec, d);
        if (pv && typeof pv.position !== "boolean" && pv.position) {
          const ecf = sat.eciToEcf(pv.position, sat.gstime(d));
          const look = sat.ecfToLookAngles(obs, ecf);
          el = (look.elevation * 180) / Math.PI; az = (look.azimuth * 180) / Math.PI;
          if (el > 10) {
            const sv = A.GeoVector(A.Body.Sun, d, true), sn = Math.hypot(sv.x, sv.y, sv.z);
            const s = [sv.x / sn, sv.y / sn, sv.z / sn], r = pv.position;
            const dp = r.x * s[0] + r.y * s[1] + r.z * s[2];
            const perp = Math.hypot(r.x - dp * s[0], r.y - dp * s[1], r.z - dp * s[2]);
            visible = !(dp < 0 && perp < 6371);
          }
        }
      }
      if (visible) { if (!cur) cur = { s: ms, e: ms, el, az }; cur.e = ms; if (el > cur.el) { cur.el = el; cur.az = az; } }
      else if (cur) { if (cur.el >= 15) out.push({ name, startMs: cur.s, start: fmt(new Date(cur.s)), end: fmt(new Date(cur.e)), maxEl: cur.el, maxAz: cur.az }); cur = null; }
    }
  }
  return { passes: out.sort((a, b) => a.startMs - b.startMs), staleDays: stale };
}

/** One short sentence for a row, from a template. */
export const rowSentence = (r: Seen) =>
  r.group === "Satellite" ? `${r.name} crosses the sky ${r.why} local time, up to ${r.where.replace("max ", "")}. A bright moving dot, no telescope needed.`
  : r.group === "Meteors" ? `${r.name}: expect ${r.why.replace("about ", "about ")}, look wide, ${r.where}.`
  : r.group === "Moon" ? `The Moon is ${r.where}, ${r.why}.`
  : `${r.name} is ${r.verdict === "Easy" ? "easy" : r.verdict === "Possible" ? "possible" : "hard"} to see with ${r.how}: ${r.where}${r.why ? `, ${r.why}` : ""}.`;
