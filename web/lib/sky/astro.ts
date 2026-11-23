// Everything astronomical, computed in the browser for any place and date (no network, no forecast needed).
import * as A from "astronomy-engine";

export type Catalog = {
  stars: [number, number, number, string, string][]; // ra h, dec, mag, name, constellation
  cons: { id: string; name: string; ra: number; dec: number | null; lines: [number, number][][] }[];
  messier: [string, string, string, number, number, number, string][]; // id, name, type, mag, ra h, dec, size
};

export const PLANETS = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"] as const;
const DEG = Math.PI / 180;

/** Offset (ms) of a time zone from UTC at a given instant. */
export function tzOffset(ms: number, tz: string): number {
  const f = new Intl.DateTimeFormat("en-US", { timeZone: tz, hourCycle: "h23", year: "numeric", month: "numeric", day: "numeric", hour: "numeric", minute: "numeric", second: "numeric" });
  const p = Object.fromEntries(f.formatToParts(new Date(ms)).map((x) => [x.type, +x.value]));
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second) - Math.floor(ms / 1000) * 1000;
}

export function localToUtc(y: number, m: number, d: number, h: number, tz: string): Date {
  const want = Date.UTC(y, m - 1, d, h);
  let t = want;
  for (let i = 0; i < 3; i++) t = want - tzOffset(t, tz);
  return new Date(t);
}

export const zoneFor = (lat: number, lon: number) => (lat < 31 && lon < -12 ? "Atlantic/Canary" : "Europe/Madrid");

export function fmtLocal(d: Date, tz: string, opts: Intl.DateTimeFormatOptions = { hour: "2-digit", minute: "2-digit", hourCycle: "h23" }) {
  return new Intl.DateTimeFormat("en-GB", { timeZone: tz, ...opts }).format(d);
}

/** Altitude and azimuth (degrees) of a J2000 RA (hours) / Dec (degrees) at a place and time. Precession ignored (< 0.4 deg). */
export function altAz(raH: number, decD: number, lat: number, lon: number, gst: number) {
  const ha = ((gst + lon / 15 - raH) * 15) * DEG;
  const dec = decD * DEG, la = lat * DEG;
  const sinAlt = Math.sin(la) * Math.sin(dec) + Math.cos(la) * Math.cos(dec) * Math.cos(ha);
  const alt = Math.asin(sinAlt);
  const az = Math.atan2(-Math.sin(ha) * Math.cos(dec), Math.cos(la) * Math.sin(dec) - Math.sin(la) * Math.cos(dec) * Math.cos(ha));
  return { alt: alt / DEG, az: ((az / DEG) + 360) % 360 };
}

const sepDeg = (r1: number, d1: number, r2: number, d2: number) => {
  const a = r1 * 15 * DEG, b = r2 * 15 * DEG, x = d1 * DEG, y = d2 * DEG;
  return Math.acos(Math.min(1, Math.sin(x) * Math.sin(y) + Math.cos(x) * Math.cos(y) * Math.cos(a - b))) / DEG;
};

export function moonName(phaseDeg: number) {
  const p = ((phaseDeg % 360) + 360) % 360;
  if (p < 22.5 || p >= 337.5) return "New Moon";
  if (p < 67.5) return "Waxing crescent";
  if (p < 112.5) return "First quarter";
  if (p < 157.5) return "Waxing gibbous";
  if (p < 202.5) return "Full Moon";
  if (p < 247.5) return "Waning gibbous";
  if (p < 292.5) return "Last quarter";
  return "Waning crescent";
}

export type Showers = { name: string; peak: [number, number]; zhr: number; ra: number; dec: number; halfWindow: number; note: string }[];
// Typical (IMO) peak dates and rates. Peaks shift by about a day from year to year; the rate is an ideal maximum under a dark sky.
export const SHOWERS: Showers = [
  { name: "Quadrantids", peak: [1, 3], zhr: 80, ra: 15.33, dec: 49.5, halfWindow: 3, note: "short, sharp peak" },
  { name: "Lyrids", peak: [4, 22], zhr: 18, ra: 18.07, dec: 34, halfWindow: 4, note: "occasional bright fireballs" },
  { name: "Eta Aquariids", peak: [5, 6], zhr: 50, ra: 22.53, dec: -1, halfWindow: 8, note: "best before dawn; debris of Halley's comet" },
  { name: "Southern Delta Aquariids", peak: [7, 30], zhr: 25, ra: 22.6, dec: -16, halfWindow: 10, note: "faint, long-lasting" },
  { name: "Perseids", peak: [8, 12], zhr: 100, ra: 3.2, dec: 58, halfWindow: 8, note: "the summer favourite; bright and many" },
  { name: "Orionids", peak: [10, 21], zhr: 20, ra: 6.33, dec: 16, halfWindow: 7, note: "fast; debris of Halley's comet" },
  { name: "Southern Taurids", peak: [11, 5], zhr: 5, ra: 3.47, dec: 13, halfWindow: 12, note: "few, slow, often bright" },
  { name: "Leonids", peak: [11, 17], zhr: 15, ra: 10.2, dec: 22, halfWindow: 4, note: "very fast" },
  { name: "Geminids", peak: [12, 14], zhr: 150, ra: 7.47, dec: 33, halfWindow: 6, note: "the strongest of the year; bright and colourful" },
  { name: "Ursids", peak: [12, 22], zhr: 10, ra: 14.47, dec: 76, halfWindow: 3, note: "low rate; best from dark sites" },
];

export type Hit = { name: string; kind: string; mag?: number; alt: number; az: number; note?: string; sub?: string };

export type Report = {
  when: Date;
  sunAlt: number;
  darkFrom?: Date; darkTo?: Date; // astronomical night (Sun below -18 deg)
  moon: { illum: number; phase: string; alt: number; az: number; rise?: Date; set?: Date; ra: number; dec: number };
  planets: (Hit & { elong: number; rise?: Date; set?: Date })[];
  stars: Hit[];
  cons: Hit[];
  deep: Hit[];
  milkyWay: { coreAlt: number; coreAz: number; verdict: string };
  showers: { name: string; zhr: number; peak: string; radiantAlt: number; daysFromPeak: number; moonIllum: number; note: string; verdict: string }[];
  chart: { stars: { x: number; y: number; m: number; n: string }[]; lines: [number, number, number, number][]; objs: { x: number; y: number; label: string; kind: string; r: number }[] };
};

function proj(alt: number, az: number) {
  // azimuthal equidistant, zenith at centre, north up, east to the left (as when looking up)
  const r = (90 - alt) / 90;
  const a = az * DEG;
  return { x: -r * Math.sin(a), y: -r * Math.cos(a) };
}

export function report(cat: Catalog, lat: number, lon: number, elev: number, when: Date): Report {
  const obs = new A.Observer(lat, lon, elev);
  const gst = A.SiderealTime(when);
  const sun = A.Equator(A.Body.Sun, when, obs, true, true);
  const sunH = A.Horizon(when, obs, sun.ra, sun.dec, "normal");
  // astronomical night around this instant
  let darkFrom: Date | undefined, darkTo: Date | undefined;
  try {
    const start = new Date(when.getTime() - 14 * 3600e3);
    const dusk = A.SearchAltitude(A.Body.Sun, obs, -1, start, 1, -18);
    const dawn = dusk ? A.SearchAltitude(A.Body.Sun, obs, +1, dusk.date, 1, -18) : null;
    if (dusk && dawn) { darkFrom = dusk.date; darkTo = dawn.date; }
    if (!dusk) { const d2 = A.SearchAltitude(A.Body.Sun, obs, +1, start, 1, -18); if (d2) darkTo = d2.date; }
  } catch { /* polar or no astronomical night */ }

  const mEq = A.Equator(A.Body.Moon, when, obs, true, true);
  const mH = A.Horizon(when, obs, mEq.ra, mEq.dec, "normal");
  const mIll = A.Illumination(A.Body.Moon, when);
  const phase = A.MoonPhase(when);
  const day = new Date(when.getTime() - 12 * 3600e3);
  const mrise = A.SearchRiseSet(A.Body.Moon, obs, +1, day, 1)?.date;
  const mset = A.SearchRiseSet(A.Body.Moon, obs, -1, day, 1)?.date;
  const moon = { illum: mIll.phase_fraction, phase: moonName(phase), alt: mH.altitude, az: mH.azimuth, rise: mrise, set: mset, ra: mEq.ra, dec: mEq.dec };

  const planets = PLANETS.map((p) => {
    const eq = A.Equator(p as A.Body, when, obs, true, true);
    const h = A.Horizon(when, obs, eq.ra, eq.dec, "normal");
    const ill = A.Illumination(p as A.Body, when);
    const el = A.Elongation(p as A.Body, when);
    const rise = A.SearchRiseSet(p as A.Body, obs, +1, day, 1)?.date;
    const set = A.SearchRiseSet(p as A.Body, obs, -1, day, 1)?.date;
    const naked = ill.mag < 6;
    return { name: p, kind: "planet", mag: ill.mag, alt: h.altitude, az: h.azimuth, elong: el.elongation, rise, set,
      note: naked ? (ill.mag < 0 ? "very bright, easy" : "naked eye") : "needs a telescope", ra: eq.ra, dec: eq.dec };
  });

  const pos = (ra: number, dec: number) => altAz(ra, dec, lat, lon, gst);
  const moonSep = (ra: number, dec: number) => sepDeg(ra, dec, moon.ra, moon.dec);

  const stars: Hit[] = cat.stars.filter((s) => s[3] && s[2] <= 2.6).map((s) => ({ name: s[3], kind: "star", mag: s[2], ...pos(s[0], s[1]), sub: s[4] }))
    .filter((s) => s.alt > 10).sort((a, b) => (a.mag ?? 9) - (b.mag ?? 9));
  const cons: Hit[] = cat.cons.filter((c) => c.dec !== null).map((c) => ({ name: c.name, kind: "constellation", ...pos(c.ra, c.dec as number) }))
    .filter((c) => c.alt > 20).sort((a, b) => b.alt - a.alt);
  const deep: Hit[] = cat.messier.filter((m) => m[3] <= 8.2).map((m) => {
    const p = pos(m[4], m[5]);
    return { name: `${m[0]} ${m[1]}`, kind: m[2], mag: m[3], ...p, sub: `${moonSep(m[4], m[5]).toFixed(0)}° from the Moon`, note: m[6] };
  }).filter((m) => m.alt > 25).sort((a, b) => (a.mag ?? 9) - (b.mag ?? 9));

  // Milky Way core: Galactic centre RA 17h45.7m, Dec -29.0
  const gc = pos(17.761, -29.008);
  const milkyWay = { coreAlt: gc.alt, coreAz: gc.az,
    verdict: gc.alt > 15 && sunH.altitude < -18 ? (moon.alt < 0 || moon.illum < 0.25 ? "core up and dark: good" : "core up, Moon washes it out")
      : gc.alt > 15 ? "core up, but the sky is not fully dark" : "core below the horizon; the rest of the band may be visible" };

  const doy = Date.UTC(when.getUTCFullYear(), when.getUTCMonth(), when.getUTCDate());
  const showers = SHOWERS.map((s) => {
    const peak = Date.UTC(when.getUTCFullYear(), s.peak[0] - 1, s.peak[1]);
    const dd = Math.round((doy - peak) / 86400e3);
    const p = pos(s.ra, s.dec);
    const active = Math.abs(dd) <= s.halfWindow;
    return { name: s.name, zhr: s.zhr, peak: `${s.peak[1]} ${["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][s.peak[0] - 1]}`,
      radiantAlt: p.alt, daysFromPeak: dd, moonIllum: moon.illum, note: s.note,
      verdict: !active ? "" : p.alt < 10 ? "active, radiant still low" : moon.illum > 0.6 && moon.alt > 0 ? "active, bright Moon spoils it" : "active tonight" };
  }).filter((s) => s.verdict !== "");

  // sky chart
  const chStars = cat.stars.filter((s) => s[2] <= 4.3).map((s) => ({ ...pos(s[0], s[1]), m: s[2], n: s[3] })).filter((s) => s.alt > 0)
    .map((s) => ({ ...proj(s.alt, s.az), m: s.m, n: s.n }));
  const lines: [number, number, number, number][] = [];
  for (const c of cat.cons) for (const seg of c.lines) for (let i = 0; i + 1 < seg.length; i++) {
    const a = pos(seg[i][0], seg[i][1]), b = pos(seg[i + 1][0], seg[i + 1][1]);
    if (a.alt > 0 && b.alt > 0) { const p = proj(a.alt, a.az), q = proj(b.alt, b.az); lines.push([p.x, p.y, q.x, q.y]); }
  }
  const objs: { x: number; y: number; label: string; kind: string; r: number }[] = [];
  if (moon.alt > 0) objs.push({ ...proj(moon.alt, moon.az), label: "Moon", kind: "moon", r: 7 });
  for (const p of planets) if (p.alt > 0 && (p.mag ?? 9) < 6) objs.push({ ...proj(p.alt, p.az), label: p.name, kind: "planet", r: 4.5 });
  for (const c of cons.slice(0, 40)) { const cc = cat.cons.find((k) => k.name === c.name)!; const a = pos(cc.ra, cc.dec as number); if (a.alt > 8) objs.push({ ...proj(a.alt, a.az), label: c.name, kind: "con", r: 0 }); }

  return { when, sunAlt: sunH.altitude, darkFrom, darkTo, moon, planets, stars, cons, deep, milkyWay, showers, chart: { stars: chStars, lines, objs } };
}

export type Ev = { date: Date; title: string; detail: string; kind: string };

/** What happens in the sky from `from` for `days` days at this place: Moon phases, planet events, eclipses, showers. */
export function upcoming(lat: number, lon: number, elev: number, from: Date, days: number): Ev[] {
  const obs = new A.Observer(lat, lon, elev);
  const end = from.getTime() + days * 86400e3;
  const out: Ev[] = [];
  // moon phases
  let q = A.SearchMoonQuarter(from);
  for (let i = 0; i < Math.ceil(days / 7) + 2; i++) {
    if (q.time.date.getTime() > end) break;
    out.push({ date: q.time.date, title: ["New Moon", "First quarter", "Full Moon", "Last quarter"][q.quarter], detail: q.quarter === 0 ? "darkest skies of the month" : q.quarter === 2 ? "bright Moon, poor for faint objects" : "", kind: "moon" });
    q = A.NextMoonQuarter(q);
  }
  // planets: oppositions and greatest elongations
  for (const p of ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune"] as A.Body[]) {
    try { const o = A.SearchRelativeLongitude(p, 0, from); if (o.date.getTime() <= end) out.push({ date: o.date, title: `${p} at opposition`, detail: "closest and brightest of the year, up all night", kind: "planet" }); } catch {}
  }
  for (const p of ["Mercury", "Venus"] as A.Body[]) {
    try { const e = A.SearchMaxElongation(p, from); if (e.time.date.getTime() <= end) out.push({ date: e.time.date, title: `${p} at greatest elongation`, detail: `${e.elongation.toFixed(0)}° from the Sun, ${e.visibility === "evening" ? "evening sky" : "morning sky"}`, kind: "planet" }); } catch {}
  }
  // close pairs: Moon-planet and planet-planet, scanned every 12 h
  const names = ["Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"] as A.Body[];
  const seen = new Set<string>();
  for (let t = from.getTime(); t < end; t += 6 * 3600e3) {
    const d = new Date(t);
    const eq = names.map((n) => A.Equator(n, d, obs, true, true));
    for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {
      const s = sepDeg(eq[i].ra, eq[i].dec, eq[j].ra, eq[j].dec);
      const lim = names[i] === "Moon" ? 2.5 : 1.5;
      if (s < lim) {
        const key = `${names[i]}-${names[j]}-${Math.floor(t / (3 * 86400e3))}`;
        if (!seen.has(key)) { seen.add(key); const sunE = A.Equator(A.Body.Sun, d, obs, true, true); const sp = sepDeg(eq[i].ra, eq[i].dec, sunE.ra, sunE.dec);
          if (sp > 20) out.push({ date: d, title: `${names[i]} close to ${names[j]}`, detail: `${s.toFixed(1)}° apart`, kind: "pair" }); }
      }
    }
  }
  // eclipses
  try { let le = A.SearchLunarEclipse(from); for (let k = 0; k < 3 && le.peak.date.getTime() <= end; k++) { out.push({ date: le.peak.date, title: `${le.kind[0].toUpperCase()}${le.kind.slice(1)} lunar eclipse`, detail: "check the Moon is above the horizon at the time", kind: "eclipse" }); le = A.NextLunarEclipse(le.peak); } } catch {}
  try { let se = A.SearchLocalSolarEclipse(from, obs); for (let k = 0; k < 3 && se.peak.time.date.getTime() <= end; k++) { out.push({ date: se.peak.time.date, title: `${se.kind[0].toUpperCase()}${se.kind.slice(1)} solar eclipse from here`, detail: `Sun ${se.peak.altitude.toFixed(0)}° high at the peak. Use eclipse glasses`, kind: "eclipse" }); se = A.NextLocalSolarEclipse(se.peak.time, obs); } } catch {}
  // meteor showers
  for (const s of SHOWERS) for (const y of [from.getUTCFullYear(), from.getUTCFullYear() + 1]) {
    const pk = new Date(Date.UTC(y, s.peak[0] - 1, s.peak[1], 3));
    if (pk.getTime() >= from.getTime() && pk.getTime() <= end) out.push({ date: pk, title: `${s.name} peak`, detail: `up to ${s.zhr} an hour under a dark sky`, kind: "shower" });
  }
  // seasons
  for (const y of [from.getUTCFullYear(), from.getUTCFullYear() + 1]) {
    const s = A.Seasons(y);
    for (const [n, t] of [["March equinox", s.mar_equinox], ["June solstice", s.jun_solstice], ["September equinox", s.sep_equinox], ["December solstice", s.dec_solstice]] as [string, A.FlexibleDateTime][]) {
      const d = (t as A.AstroTime).date; if (d.getTime() >= from.getTime() && d.getTime() <= end) out.push({ date: d, title: n, detail: "", kind: "season" });
    }
  }
  return out.sort((a, b) => a.date.getTime() - b.date.getTime());
}

/** Moon altitude and illumination at an instant, for ranking nights. */
export function moonAt(lat: number, lon: number, elev: number, when: Date) {
  const obs = new A.Observer(lat, lon, elev);
  const eq = A.Equator(A.Body.Moon, when, obs, true, true);
  const h = A.Horizon(when, obs, eq.ra, eq.dec, "normal");
  return { alt: h.altitude, illum: A.Illumination(A.Body.Moon, when).phase_fraction };
}
