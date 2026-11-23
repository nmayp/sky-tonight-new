"use client";

import { useEffect, useState } from "react";
import { hasWebGpu, MODEL, rewrite } from "@/lib/sky/llm";

type Out = { text: string; by: "rules" | "model" | "rejected"; why?: string };

/** The answer as one sentence: from fixed rules, or written by a small open model when the visitor turns it on. */
export default function Headline({ rule, pClear, names }: { rule: string; pClear: number; names: string[] }) {
  const [gpu, setGpu] = useState<boolean | null>(null);
  const [on, setOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState("");
  const [out, setOut] = useState<Out | null>(null);
  const [err, setErr] = useState("");
  const nk = names.join("|");

  useEffect(() => { hasWebGpu().then(setGpu); setOn(localStorage.getItem("sky-llm") === "1"); }, []);
  useEffect(() => {
    if (!on || !gpu) { setOut(null); return; }
    let dead = false;
    const t = setTimeout(async () => {
      setBusy(true); setErr("");
      try {
        const r = await rewrite(rule, pClear, names, (_p, note) => !dead && setProg(note));
        if (dead) return;
        setProg("");
        setOut(r.text ? { text: r.text, by: "model" } : { text: rule, by: "rejected", why: r.why ?? "" });
      } catch (e) { if (!dead) { setErr(String(e).slice(0, 120)); setOut(null); setProg(""); } }
      if (!dead) setBusy(false);
    }, 500);
    return () => { dead = true; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, gpu, rule, pClear, nk]);

  const shown = out && !busy ? out : { text: rule, by: "rules" as const };
  const toggle = () => { const v = !on; setOn(v); localStorage.setItem("sky-llm", v ? "1" : "0"); };
  return (
    <div className="sky-answer">
      <p className="sky-sentence" aria-live="polite">{shown.text}</p>
      <p className="note">
        {busy ? prog || "Writing with the open model…"
          : shown.by === "model" ? `Written by ${MODEL.split("/")[1]} in your browser, checked against the rules.`
          : shown.by === "rejected" ? `The open model's sentence failed a check (${out?.why}); this one is from fixed rules.`
          : "Written by fixed rules."}
        {gpu === true && !busy && <button type="button" className="see-more sky-inline" onClick={toggle}>{on ? "use fixed rules" : "use the open model (1.8 GB, once)"}</button>}
        {gpu === false && " The open model needs WebGPU."}
        {err && ` The open model could not run here (${err}).`}
      </p>
    </div>
  );
}
