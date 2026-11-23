import SkyPlanner from "@/components/SkyPlanner";

export default function Page() {
  return (
    <main>
      <h1>Sky Tonight</h1>
      <p className="intro">
        For any place in Spain and any night up to about six months ahead: the chance of a clear sky, and what you can see
        (Moon, planets, stars, satellites), dimmed by city light.
      </p>
      <SkyPlanner />
      <footer className="note">
        Forecasts: NOAA GFS. Cloud reports: SYNOP stations. Night lights: NASA VIIRS Black Marble.
      </footer>
    </main>
  );
}
