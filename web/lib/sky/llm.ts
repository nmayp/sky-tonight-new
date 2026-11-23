// Optional feature: a small open model that runs in the browser (WebGPU) rewrites the rule-based facts as one short friendly
// sentence. Nothing is sent to a server. The rule-based sentence stays the source of truth, and the model's output is thrown
// away if it changes a number or contradicts the chance of a clear sky.

/** The model needs WebGPU: on the processor alone it runs out of memory. */
export async function hasWebGpu() {
  try { const gpu = (navigator as unknown as { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu; return !!gpu && !!(await gpu.requestAdapter()); } catch { return false; }
}

export const MODEL = "onnx-community/Qwen2.5-1.5B-Instruct";
const LIB = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

const SYSTEM =
  "Rewrite the sentence in simpler, friendlier words. Keep it to ONE sentence. Keep every number and every name exactly. Do not add anything.";

type Gen = (msgs: { role: string; content: string }[], o: object) => Promise<{ generated_text: { content: string }[] }[]>;
let pipe: Promise<Gen> | null = null;

export function loadModel(onProgress: (pct: number, note: string) => void): Promise<Gen> {
  if (!pipe) {
    pipe = (async () => {
      const tf = await import(/* webpackIgnore: true */ /* @vite-ignore */ LIB);
      const dev = "webgpu";
      onProgress(0, "Downloading the model (about 1.8 GB), using your graphics card");
      const files: Record<string, { loaded: number; total: number }> = {};
      return (await tf.pipeline("text-generation", MODEL, {
        dtype: "q4", device: dev,
        progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
          if (p.status === "progress" && p.file && p.total) {
            files[p.file] = { loaded: p.loaded ?? 0, total: p.total };
            const l = Object.values(files).reduce((a, b) => a + b.loaded, 0), t = Object.values(files).reduce((a, b) => a + b.total, 0);
            onProgress(Math.round((l / t) * 100), `Downloading the model: ${Math.round(l / 1e6)} of ${Math.round(t / 1e6)} MB`);
          }
        },
      })) as unknown as Gen;
    })().catch((e) => { pipe = null; throw e; });
  }
  return pipe;
}

const nums = (x: string) => (x.match(/\d+(?:[.,]\d+)?/g) ?? []).map((n) => n.replace(",", "."));
const FORBID = /(sunshine|sunny|rain|storm|forecast says|washed out by city|you can still see|guarantee)/i;

/** Accept the model's sentence only if it keeps the numbers and the meaning of the rule sentence. */
export function check(out: string, rule: string, pClear: number, names: string[]): string | null {
  const allowed = new Set(nums(rule));
  for (const n of nums(out)) if (!allowed.has(n)) return `it introduced the number ${n}`;
  const low = out.toLowerCase();
  if (!/clear/.test(low)) return "it did not mention the chance of a clear sky";
  if (!nums(out).includes(String(Math.round(pClear * 100)))) return "it dropped the percentage";
  if (pClear < 0.4 && !/cloud/.test(low)) return "it did not say the sky will probably be cloudy";
  if (pClear >= 0.6 && /(cloudy|clouds)/.test(low) && !/(little|few|not|unlikely)/.test(low)) return "it called a likely clear night cloudy";
  if (FORBID.test(out)) return "it added something the facts do not say";
  const keep = names.slice(0, 3).filter((n) => !low.includes(n.toLowerCase()));
  if (keep.length > 0) return `it dropped ${keep[0]}`;
  if (out.trim().length < 20) return "the answer was empty";
  return null;
}

const FIRST = /(?<=[a-z\)])[.!?]\s+(?=[A-Z])/;

/** Ask the model for a friendlier version of the rule sentence; up to three tries, each checked. */
export async function rewrite(rule: string, pClear: number, names: string[], onProgress: (pct: number, note: string) => void) {
  const gen = await loadModel(onProgress);
  const shots = [["Girona is mostly cloudy tonight (12 % chance of a clear sky), and if it clears, the Moon and Jupiter are easy to see.", "Girona will probably be cloudy tonight, with only a 12 % chance of a clear sky, but the Moon and Jupiter are easy to spot if it clears."]];
  const msgs: { role: string; content: string }[] = [{ role: "system", content: SYSTEM }];
  for (const [u, a] of shots) msgs.push({ role: "user", content: u }, { role: "assistant", content: a });
  msgs.push({ role: "user", content: rule });
  let last = "";
  for (let k = 0; k < 3; k++) {
    const out = await gen(msgs, k === 0 ? { max_new_tokens: 70, do_sample: false } : { max_new_tokens: 70, do_sample: true, temperature: 0.7, top_p: 0.9 });
    let t = out[0].generated_text.at(-1)!.content.trim().split(FIRST)[0].trim();
    if (!/[.!?]$/.test(t)) t += ".";
    last = t;
    if (!check(t, rule, pClear, names)) return { text: t, tries: k + 1, why: null as string | null };
  }
  return { text: null as string | null, tries: 3, why: check(last, rule, pClear, names) };
}
