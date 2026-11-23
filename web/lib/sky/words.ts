// Plain-language sentences from numbers. Fixed bands and templates: no language model, the same input always gives the same sentence.

export const CHANCE_BANDS: [number, string, string][] = [
  [0.1, "Under 10 %", "Almost certainly cloudy. Do not plan around it."],
  [0.25, "10–25 %", "Mostly cloudy. About 1 night in 5 is clear."],
  [0.4, "25–40 %", "Often cloudy. About 3 nights in 10 are clear."],
  [0.6, "40–60 %", "A coin flip. About half of such nights are clear."],
  [0.75, "60–75 %", "Likely clear. About 2 nights in 3."],
  [0.9, "75–90 %", "Very likely clear. Only 1 night in 5 or fewer is spoiled."],
  [1.01, "Over 90 %", "Almost always clear."],
];

export const chanceWords = (p: number) => CHANCE_BANDS.find((b) => p < b[0])![2];

export const sureWords = (w: number, hasForecast: boolean, leadH: number) =>
  !hasForecast ? "Not a forecast: the usual weather for this time of year, because the stored forecast does not reach this date."
  : w >= 0.8 ? `A real forecast (${Math.round(leadH)} hours ahead). Trust it fairly well.`
  : w >= 0.4 ? `Part forecast, part usual weather (${Math.round(leadH / 24)} days ahead). Check again nearer the day.`
  : `Mostly the usual weather (${Math.round(leadH / 24)} days ahead). The forecast adds little this far out.`;

export const DARK_WORDS: Record<string, string> = {
  "very dark": "Very dark sky. The Milky Way and faint objects are easy.",
  dark: "Dark sky. The Milky Way shows on a clear moonless night.",
  moderate: "Suburban sky. Planets, bright stars and the Moon are fine; faint objects are hard.",
  bright: "Bright sky. Only planets, the Moon and the brightest stars.",
  "very bright": "City sky. Only the Moon, bright planets and a few stars.",
};

export const moonWords = (up: boolean, illum: number, phase: string) =>
  !up ? `The Moon is down (${phase.toLowerCase()}), so faint objects are not washed out.`
  : illum > 0.7 ? `A bright Moon is up (${Math.round(illum * 100)} % lit). It washes out everything faint.`
  : illum > 0.3 ? `The Moon is up (${Math.round(illum * 100)} % lit). It hides the faintest objects.`
  : `A thin Moon is up (${Math.round(illum * 100)} % lit). It barely matters.`;

// Rough counts of stars a person can see at once above the horizon for a faintest visible magnitude.
const STARS: [number, number][] = [[2.5, 40], [3.5, 150], [4.5, 450], [5.5, 1300], [6.2, 2400]];
export const starsWords = (lm: number) => {
  if (lm < 1) return "Almost no stars are visible now.";
  const n = lm >= STARS[STARS.length - 1][0] ? STARS[STARS.length - 1][1] : (STARS.find((s) => lm <= s[0]) ?? STARS[0])[1];
  return `The faintest star you can see is about magnitude ${lm.toFixed(1)}, so roughly ${n.toLocaleString("en-GB")} stars at once.`;
};

export const overallWords = (p: number, lm: number, moonSpoils: boolean, isNight: boolean) => {
  if (!isNight) return p >= 0.5 ? "Not dark enough yet. Wait for night." : "Not dark enough yet, and probably cloudy.";
  if (lm < 2.5) return p >= 0.25 ? "Only the Moon, bright planets and a few stars: city light and the Moon wash out the rest." : "Poor night: probably cloudy, and the sky is washed out anyway.";
  if (p < 0.25) return "Poor night: probably cloudy.";
  if (lm < 4) return p >= 0.5 ? "Planets and the Moon only: the sky is too bright here for much else." : "Poor night.";
  if (moonSpoils) return p >= 0.5 ? "Good for the Moon and planets; faint objects are washed out." : "Poor night, and the Moon is bright.";
  return p >= 0.6 ? "Good night for stargazing." : "Worth a try if you can be flexible.";
};

export const verdictWord = (v: string) => (v === "Easy" ? "easy" : v === "Possible" ? "possible" : "hard");

const BAND = (p: number) => p < 0.1 ? "it will almost certainly be cloudy" : p < 0.25 ? "it will probably be cloudy" : p < 0.4 ? "it is often cloudy" : p < 0.6 ? "it is a coin flip" : p < 0.75 ? "it is likely clear" : p < 0.9 ? "it is very likely clear" : "it is almost always clear";
const list = (a: string[]) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);

/** The whole answer as one sentence, from fixed rules. */
export function headline(o: { place: string; when: string; p: number; isNight: boolean; lm: number; moonBright: boolean; easy: string[]; possible: string[] }) {
  const pc = Math.round(o.p * 100);
  const art = /^(8|11|18)/.test(String(pc)) ? "an" : "a";
  const head = `${o.place} has ${art} ${pc} % chance of a clear sky ${o.when}, so ${BAND(o.p)}`;
  if (!o.isNight) return `${head}, but it is not dark enough yet to see much.`;
  const easy = o.easy.slice(0, 3);
  if (easy.length === 0) return `${head}, and nothing is easy to see${o.moonBright ? " under a bright Moon" : o.lm < 3.5 ? " in this city light" : ""}.`;
  const cond = o.p < 0.4 ? "if it clears, " : "";
  const wash = o.moonBright && o.lm >= 3.5 ? ", though the Moon hides faint objects" : o.lm < 3.5 ? ", though city light hides the faint ones" : "";
  return `${head}, and ${cond}${list(easy)} ${easy.length === 1 ? "is" : "are"} easy to see${wash}.`;
}
