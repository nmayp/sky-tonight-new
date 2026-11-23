"use client";

import { useEffect, useMemo, useState } from "react";
import Headline from "./Headline";
import Flow from "./Flow";
import { SkyLead, SkyScore } from "./SkyResults";
import MiniChart from "./MiniChart";
import { ClearSky, type Forecast, type Grid, type Lead, type Lights } from "@/lib/sky/clear";
import { fmtLocal, localToUtc, report, zoneFor, type Catalog } from "@/lib/sky/astro";
import { DARK_FACTOR, satellitePasses, whatYouSee, type Tle } from "@/lib/sky/visible";
import { headline } from "@/lib/sky/words";

type PlacesFile = { regions: string[]; places: [string, number, number, number, number, number][] };
type Loaded = { cat: Catalog; sky: ClearSky; places: PlacesFile; tle: Tle; trainedThrough: string };
type Spot = { name: string; sub: string; lat: number; lon: number; elev: number };

const MAX_DAYS = 180;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const dayLabel = (d: Date) => `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
const norm = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
const km = (a: { lat: number; lon: number }, la: number, lo: number) => {
  const r = Math.PI / 180;
  return 6371 * Math.hypot((lo - a.lon) * r * Math.cos(((la + a.lat) / 2) * r), (la - a.lat) * r);
};

export default function SkyPlanner() {
  const [data, setData] = useState<Loaded | null>(null);
  const [err, setErr] = useState("");
  const [spot, setSpot] = useState<Spot>({ name: "Barcelona", sub: "Catalonia", lat: 41.389, lon: 2.159, elev: 47 });
  const [ahead, setAhead] = useState(0);
  const [hour, setHour] = useState(5);
  const [q, setQ] = useState<string | null>(null);
  const today = useMemo(() => { const n = new Date(); return Date.UTC(n.getFullYear(), n.getMonth(), n.getDate()); }, []);

  useEffect(() => {
    const j = (f: string) => fetch(`/sky/${f}`).then((r) => { if (!r.ok) throw new Error(f); return r.json(); });
    Promise.all([j("catalog.json"), j("grid.json"), j("forecast_v2.json"), j("skill.json"), j("nightlights.json"), j("places.json"), j("tle.json")])
      .then(([cat, grid, fc, skill, lights, places, tle]) =>
        setData({ cat, places, tle, sky: new ClearSky(grid as Grid, fc as Forecast, skill.lead as Lead[], lights as Lights), trainedThrough: skill.trained_through }))
      .catch((e) => setErr(String(e)));
  }, []);

  const tz = zoneFor(spot.lat, spot.lon);
  const night = useMemo(() => new Date(today + ahead * 86400e3), [today, ahead]);
  const h24 = 19 + hour;
  const when = useMemo(() => localToUtc(night.getUTCFullYear(), night.getUTCMonth() + 1, night.getUTCDate() + (h24 >= 24 ? 1 : 0), h24 % 24, tz), [night, h24, tz]);
  const matches = useMemo(() => {
    if (!data || !q || q.length < 2) return [];
    const n = norm(q);
    return data.places.places.filter((p) => norm(p[0]).includes(n)).slice(0, 5);
  }, [data, q]);

  const res = useMemo(() => (data ? data.sky.chance(spot.lat, spot.lon, when) : null), [data, spot, when]);
  const rep = useMemo(() => (data ? report(data.cat, spot.lat, spot.lon, spot.elev, when) : null), [data, spot, when]);
  const dark = useMemo(() => (data ? data.sky.darkness(spot.lat, spot.lon) : null), [data, spot]);
  const sats = useMemo(() => {
    if (!data) return null;
    const from = localToUtc(night.getUTCFullYear(), night.getUTCMonth() + 1, night.getUTCDate(), 18, tz).getTime();
    return satellitePasses(data.tle, spot.lat, spot.lon, spot.elev, from, from + 13 * 3600e3, (d) => fmtLocal(d, tz));
  }, [data, spot, night, tz]);
  const seen = useMemo(() => (rep && dark ? whatYouSee(rep, dark.cls, sats && sats.staleDays < 14 ? sats.passes : []) : null), [rep, dark, sats]);

  const where = useMemo(() => {
    if (!data) return [];
    return data.places.places.filter((p) => p[3] >= 300).map((p) => {
      const c = data.sky.chance(p[1], p[2], when).p, d = data.sky.darkness(p[1], p[2]);
      return { name: p[0], lat: p[1], lon: p[2], c, d, score: c * (DARK_FACTOR[d.cls] ?? 0.5) };
    });
  }, [data, when]);
  const nearby = useMemo(() => where.map((r) => ({ ...r, dist: km(spot, r.lat, r.lon) })).filter((r) => r.dist <= 120 && r.dist > 3).sort((a, b) => b.score - a.score).slice(0, 6), [where, spot]);
  const spain = useMemo(() => where.slice().sort((a, b) => b.score - a.score).slice(0, 6), [where]);

  const series = useMemo(() => {
    if (!data) return null;
    const ch: [number, number][] = [], cl: [number, number][] = [], wt: [number, number][] = [];
    for (let d = 0; d <= MAX_DAYS; d += 2) {
      const nd = new Date(today + d * 86400e3);
      const t = localToUtc(nd.getUTCFullYear(), nd.getUTCMonth() + 1, nd.getUTCDate() + (h24 >= 24 ? 1 : 0), h24 % 24, tz);
      const r = data.sky.chance(spot.lat, spot.lon, t);
      ch.push([d, r.p * 100]); cl.push([d, r.clim * 100]); wt.push([d, r.w * 100]);
    }
    return { ch, cl, wt };
  }, [data, spot, h24, tz, today]);

  if (err) return <p className="note">Could not load the sky data ({err}).</p>;
  if (!data || !res || !rep || !seen || !dark || !series) return <p className="note">Loading…</p>;

  const short = (r: { group: string; name: string }) =>
    r.group === "Moon" ? "the Moon" : r.group === "Meteors" ? `the ${r.name.replace(" meteor shower", "")}` : r.name === "Milky Way core" ? "the Milky Way" : r.name.replace(/ \(.*\)/, "").replace(/^M\d+ (NGC \d+ )?/, "");
  const easy = seen.rows.filter((r) => r.verdict === "Easy").map(short);
  const poss = seen.rows.filter((r) => r.verdict === "Possible" && r.group !== "Deep sky").map(short);
  const moonSpoils = rep.moon.alt > 0 && rep.moon.illum > 0.5;
  const whenStr = `${ahead === 0 ? "tonight" : `on ${dayLabel(night)}`} at ${String(h24 % 24).padStart(2, "0")}:00`;
  const rule = headline({ place: spot.name, when: whenStr, p: res.p, isNight: rep.sunAlt < -12, lm: seen.lm, moonBright: moonSpoils, easy, possible: poss });
  const dateVal = night.toISOString().slice(0, 10);
  const fcEnds = series.wt.find((p) => p[1] < 5);
  const satNote = sats && sats.staleDays >= 14 ? "Satellite orbits are only known for about two weeks, so no passes are shown for this date." : "";

  const best = nearby[0];
  const top = spain[0];
  const nice = (r: { group: string; name: string }) => short(r).replace(/^the /, "").replace(/^./, (c) => c.toUpperCase());
  const controls = (
    <div className="sky-ctl">
      <Headline rule={rule} pClear={res.p} names={easy} />
      <div className="pair">
        <div>
          <label htmlFor="sky-place">Place in Spain</label>
          <input id="sky-place" value={q === null ? spot.name : q} onFocus={(e) => e.currentTarget.select()} onChange={(e) => setQ(e.target.value)} placeholder="Type a town or village" />
          {matches.length > 0 && (
            <ul className="sky-pick">
              {matches.map((p, i) => (
                <li key={i}><button type="button" className="see-more" onClick={() => { setSpot({ name: p[0], sub: data.places.regions[p[4]] ?? "", lat: p[1], lon: p[2], elev: p[5] }); setQ(null); }}>{p[0]}</button> <span className="note">{data.places.regions[p[4]]}</span></li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <label htmlFor="sky-date">Night</label>
          <input id="sky-date" type="date" value={dateVal} min={new Date(today).toISOString().slice(0, 10)} max={new Date(today + MAX_DAYS * 86400e3).toISOString().slice(0, 10)}
            onChange={(e) => { const v = Date.parse(e.target.value); if (!Number.isNaN(v)) setAhead(Math.max(0, Math.min(MAX_DAYS, Math.round((v - today) / 86400e3)))); }} />
        </div>
      </div>
      <label>Hour</label>
      <div className="sky-chips">
        {Array.from({ length: 13 }, (_, i) => (19 + i) % 24).map((h, i) => (
          <button type="button" className="see-more" key={i} aria-pressed={hour === i} onClick={() => setHour(i)}>{String(h).padStart(2, "0")}</button>
        ))}
      </div>
    </div>
  );


  return (
    <>
      <section id="tonight">
        <h2>Tonight</h2>
        {controls}
      </section>

      <section id="see">
        <h2>What you can see</h2>
        <table>
          <tbody>
            {seen.rows.slice(0, 10).map((r, i) => <tr key={i}><th scope="row">{nice(r)}</th><td>{r.where}</td><td>{r.verdict.toLowerCase()}</td></tr>)}
            {seen.rows.length === 0 && <tr><td>Not dark enough for anything worth looking for.</td></tr>}
          </tbody>
        </table>
        <p className="note">At {spot.name}, if the sky is clear. {satNote}</p>
      </section>

      <section id="where">
        <h2>Where to go</h2>
        <p className="lead small">{top ? `Best in Spain that night: ${top.name}, ${Math.round(top.c * 100)} % clear, ${top.d.cls} sky.` : ""}</p>
        <div className="pair">
          <div>
            <p className="cap">Within 120 km of {spot.name}</p>
            <table>
              <tbody>{nearby.slice(0, 6).map((r) => <tr key={r.name}><th scope="row">{r.name}</th><td>{Math.round(r.c * 100)} %</td><td>{r.d.cls}</td></tr>)}</tbody>
            </table>
          </div>
          <div>
            <p className="cap">Best in Spain</p>
            <table>
              <tbody>{spain.slice(0, 6).map((r) => <tr key={r.name}><th scope="row">{r.name}</th><td>{Math.round(r.c * 100)} %</td><td>{r.d.cls}</td></tr>)}</tbody>
            </table>
          </div>
        </div>
        <p className="note">{best ? "Chance of a clear sky and the darkness of the sky, for the hour you picked." : `Nothing within 120 km of ${spot.name}.`}</p>
      </section>

      <section id="days">
        <h2>Next 180 days</h2>
        <div className="chart">
          <MiniChart aria="Chance of a clear sky over the next 180 days, with the monthly average"
            series={[{ label: "chance", pts: series.ch, blue: true }, { label: "average", pts: series.cl, dash: "dashed" }]}
            xTicks={[0, 60, 120, 180]} xFmt={(v) => dayLabel(new Date(today + v * 86400e3))} yTicks={[0, 50, 100]} yFmt={(v) => `${v}`}
            yMin={0} yMax={100} xLabel="" yLabel="% clear" marker={ahead} />
          <p className="note">At {spot.name}. The forecast covers about {fcEnds ? fcEnds[0] : 10} days, then the line is the monthly average. Trained to {data.trainedThrough}.</p>
        </div>
      </section>

      <section id="how">
        <h2>How it works</h2>
        <Flow
          steps={[
            "Offline · Python",
            { n: "1", t: "Data in", d: "NOAA forecasts 2023–26, cloud and haze reports from 56 stations" },
            { n: "2", t: "Clear-sky model", d: "boosted trees, calibrated, tested on unseen 2026", blue: true },
            "In your browser · no server",
            { n: "3", t: "Clear chance", d: "the forecast fades into the monthly average", blue: true },
            { n: "4", t: "What is up", d: "Moon, planets, stars, dimmed by city light" },
            { n: "5", t: "One sentence", d: "fixed rules, or a small open model in the browser" },
          ]}
        />
      </section>

      <section id="works">
        <h2>Does it work?</h2>
        <SkyScore />
      </section>

      <section id="lead">
        <h2>Lead time</h2>
        <SkyLead />
      </section>

      <section id="limits">
        <h2>Limits</h2>
        <ul className="plain">
          <li>Clear is a station observer&apos;s estimate: 0–2 eighths of sky covered.</li>
          <li>Only a few of 56 stations sit at observatory height.</li>
          <li>Catalonia hold-out (2 stations): v2 worse than the plain forecast.</li>
          <li>Tenerife camera check (52 frames, 9 nights): beats the plain forecast; constant guess still wins.</li>
          <li>Forecasts cover about 10 days; after that the chance is the monthly average.</li>
          <li>Sky brightness is a model of city light, not a measurement.</li>
        </ul>
      </section>
    </>
  );
}
