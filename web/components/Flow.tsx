/** A diagram as a vertical chain of steps: numbered dots on one thin line. No boxes. */
export default function Flow({ steps }: { steps: ({ n: string; t: string; d: string; blue?: boolean } | string)[] }) {
  return (
    <ol className="flow">
      {steps.map((s, j) =>
        typeof s === "string" ? (
          <li key={j} className="grp">{s}</li>
        ) : (
          <li key={j} data-n={s.n} className={s.blue ? "blue" : undefined}>
            <b>{s.t}</b>
            <span>{s.d}</span>
          </li>
        ),
      )}
    </ol>
  );
}
