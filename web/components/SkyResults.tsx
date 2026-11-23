"use client";

import { useEffect, useState } from "react";
import MiniChart from "./MiniChart";

type Skill = {
  trained_through: string; train_through: string; test_rows: number;
  reliability_model: { p: number; obs: number; n: number }[]; reliability_raw: { p: number; obs: number; n: number }[];
  monthly: { month: number; model: number; raw: number; clim: number }[];
  overall: { model: { brier: number; auc: number; ece: number }; raw: { brier: number } };
  lead: { lead_h: number; model: number; clim: number; blend: number; best_weight: number }[];
  clim_check: { r: number; mae: number };
};

function useSkill() {
  const [s, setS] = useState<Skill | null>(null);
  useEffect(() => { fetch("/sky/skill.json").then((r) => r.json()).then(setS).catch(() => {}); }, []);
  return s;
}

export function SkyScore() {
  const s = useSkill();
  if (!s) return <p className="note">Loading…</p>;
  return (
    <>
      <p className="lead">Beats a plain cloud rule on {s.test_rows.toLocaleString("en-GB")} nights it never saw.</p>
      <ul className="tiles">
        <li><span className="n">{s.overall.model.brier.toFixed(3)}</span><span className="l">error, model</span></li>
        <li><span className="n">{s.overall.raw.brier.toFixed(3)}</span><span className="l">error, cloud rule</span></li>
      </ul>
      <p className="note">Brier score, lower is better. Station-nights of 2026; trained to {s.train_through}.</p>
    </>
  );
}

export function SkyLead() {
  const s = useSkill();
  if (!s) return <p className="note">Loading…</p>;
  return (
    <div className="chart">
      <MiniChart aria="Error against forecast lead time in days" series={[{ label: "model", pts: s.lead.map((l) => [l.lead_h / 24, l.model]), blue: true }, { label: "average", pts: s.lead.map((l) => [l.lead_h / 24, l.clim]), dash: "dashed" }]}
        xTicks={[0, 3, 6, 9]} xFmt={(v) => `${v} d`} yTicks={[0.1, 0.2, 0.3]} yFmt={(v) => v.toFixed(1)} yMin={0.1} yMax={0.3} xLabel="days ahead" yLabel="error" />
      <p className="note">After about 6 days the monthly average is as good, so the page blends the two.</p>
    </div>
  );
}
