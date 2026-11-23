"""Sky tonight: where and when you can see stars, satellites and other sky events."""
__version__ = "0.2.0"

# name -> (label, lat, lon, elevation m, IANA time zone, nearest SYNOP block or None)
SITES = {
    "barcelona": ("Barcelona (El Prat)", 41.2971, 2.0785, 4, "Europe/Madrid", "08181"),
    "montsec": ("Montsec observatory, Lleida", 42.0510, 0.7295, 1570, "Europe/Madrid", None),
    "javalambre": ("Javalambre observatory, Teruel", 40.0419, -1.0164, 1957, "Europe/Madrid", None),
    "calar_alto": ("Calar Alto observatory, Almeria", 37.2236, -2.5462, 2168, "Europe/Madrid", None),
    "sierra_nevada": ("Sierra Nevada observatory, Granada", 37.0642, -3.3847, 2896, "Europe/Madrid", None),
    "teide": ("Teide observatory, Tenerife", 28.3009, -16.5099, 2390, "Atlantic/Canary", None),
    "roque": ("Roque de los Muchachos, La Palma", 28.7569, -17.8920, 2396, "Atlantic/Canary", None),
}
# Places outside Spain used only to check the model against an observatory's all-sky camera.
CHECK_SITES = {
    "apache_point": ("Apache Point Observatory, New Mexico", 32.7803, -105.8203, 2788, "America/Denver", None),
    # AllSkyKamera ASK076 "Deep Sky Teide", Tenerife (Spain). Elevation 275 m is the camera's own overlay value.
    "deep_sky_teide": ("Deep Sky Teide all-sky camera, Tenerife", 28.2503, -16.8293, 275, "Atlantic/Canary", None),
}
DEFAULT = "barcelona"


def get_site(key: str = DEFAULT) -> dict:
    label, lat, lon, elev, tz, block = {**SITES, **CHECK_SITES}[key]
    return {"key": key, "name": label, "lat": lat, "lon": lon, "elev_m": elev, "tz": tz, "synop": block}


SITE = get_site(DEFAULT)  # kept for code that predates multi-site support
