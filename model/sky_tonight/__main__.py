import argparse
import datetime as dt
import json
import warnings

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser(prog="sky_tonight")
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("train", help="fit the clear-sky model on 2023+2025, test on 2026")
b = sub.add_parser("build", help="fetch forecast, compute events, log, write site/index.html")
b.add_argument("--date", default=None)
b.add_argument("--no-iss", action="store_true")
b.add_argument("--db", default="data/live.sqlite")
sc = sub.add_parser("score", help="score logged forecasts against SYNOP total-cloud observations")
sc.add_argument("--db", default="data/live.sqlite")
pl = sub.add_parser("planner", help="compute tonight's ranking for every place -> site/planner.json")
pl.add_argument("--date", default=None)
fs = sub.add_parser("forecasts", help="log forecasts for every place")
fs.add_argument("--db", default="data/live.sqlite")
al = sub.add_parser("allsky", help="log one image from the live all-sky camera at the GTC (night only)")
al.add_argument("--out", default="data/allsky")
al.add_argument("--force", action="store_true")
wc = sub.add_parser("webcams", help="log the ordinary webcams at the GTC (night only)")
wc.add_argument("--out", default="data/webcams")
wc.add_argument("--force", action="store_true")
sy = sub.add_parser("synop", help="download SYNOP history (Ogimet) to data/synop_08181.csv")
sy.add_argument("--from-year", type=int, default=2023)
a = p.parse_args()

if a.cmd == "train":
    from . import train
    train.main()
elif a.cmd == "build":
    from . import events, live, site
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(events.TZ).date()
    bundle = live.load_model()
    fc, source = live.fetch_forecast()
    pred = live.predict(fc, bundle)
    con = live.connect(a.db)
    n = live.log(con, pred, source)
    sc = live.scoreboard(con)
    ev = events.tonight(day, with_iss=not a.no_iss)
    html = site.render(ev, pred, sc, json.load(open("data/scores.json")))
    open("site/index.html", "w").write(html)
    print(f"forecast from {source}; logged {n} forecast hours; scoreboard: {sc}; wrote site/index.html")
elif a.cmd == "score":
    from . import live
    print(json.dumps(live.scoreboard(live.connect(a.db)), indent=2))
elif a.cmd == "planner":
    import datetime as _dt
    from . import planner
    r = planner.build(_dt.date.fromisoformat(a.date) if a.date else None)
    for p_ in r["places"]:
        print(f"{p_['name'][:38]:38} good hours {p_['expected_good_hours']:4}  {p_['darkness']}")
elif a.cmd == "forecasts":
    from . import live
    for k, v in live.log_all_sites(live.connect(a.db), live.load_model()).items():
        print(f"{k:14} {v}")
elif a.cmd == "allsky":
    from . import allsky
    print(allsky.capture(a.out, a.force))
elif a.cmd == "webcams":
    from . import allsky
    print(allsky.capture_webcams(a.out, a.force))
elif a.cmd == "synop":
    import datetime as _dt
    from . import synop
    today = _dt.date.today()
    months = [(y, m) for y in range(a.from_year, today.year + 1) for m in range(1, 13)
              if (y, m) <= (today.year, today.month)]
    print(len(synop.fetch_history(months)), "hours written")
