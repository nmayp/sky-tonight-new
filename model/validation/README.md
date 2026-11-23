# Validation frames

Two all-sky cameras are used to check the clear-sky model against what the sky really looked like. The Spanish camera
(Tenerife) is the main check. The Apache Point frames (New Mexico) were a first test and are kept only as that: the
site is far from Spain, where the model is meant to be used.

## Main check: Tenerife (AllSkyKamera ASK076 "Deep Sky Teide")

`tenerife/` holds one frame every two hours (20, 22, 00, 02, 04, 06 UT) from the camera's nightly timelapse videos for
the nights of 13-14 to 21-22 November 2026, all with the Sun more than 12 degrees below the horizon, 520 px wide, and
`tenerife/labels.json` with my reading of each (clear / partly / cloudy, plus a note and the Moon's height and
illumination). The reading is a judgement made by eye by one person, not a measurement. Clear counts 1, partly 0.5 and
cloudy 0 in the Brier score. Two frames were dropped as unreadable (a bright veil and drops on the dome, thin cloud
could not be told from Moon glare), leaving 52 frames over 9 nights: 35 clear, 8 partly, 9 cloudy.
`tenerife/make_frames.py` downloads the videos and rebuilds the frames; each frame's time is the camera's own clock
overlay (UT; checked against the Moon's rise and set), interpolated between readings on the first, middle and last
frame of each video.

Credit: AllSkyKamera Network (https://allskykamera.space), camera ASK076 Deep Sky Teide, Tenerife (Joao Berbereia),
https://allskykamera.space/kamera.php?k=ASK076. The frames are shown under the Network's FAQ default ("use with
attribution and link"). Its licence page describes the rights camera owners grant to the Network and does not itself
license third parties, so explicit permission is being requested separately.

Result (`PYTHONPATH=. python validation/score_tenerife.py`, real GFS forecasts from the 12 UT run of the night's start date, 8-18 h
ahead; nothing tuned): Brier 0.389 for the model, 0.268 for the calibrated raw forecast, 0.149 for a constant guess.
The model loses to both. On the 44 clear-or-cloudy frames it scored 13 hits (the raw forecast 23); a forecast that
always said "clear" would score 35. A hit is a probability of 50% or more (whole percent, as shown on the page) on a
clear frame, or under 50% on a cloudy one. Ranking clear against cloudy frames, the model is worse than chance (0.25,
where 0.5 is chance), the raw forecast 0.38. Using the Apache Point rule instead (18 UT run of the day before, 26-28 h ahead for
the 20 and 22 UT frames) gives 0.414 / 0.267 / 0.149. Resampling whole nights, the model's Brier is above the constant
guess by 0.11 to 0.40 (95% of resamples). The model also says 34% clear on average where I read 75%.

How far to trust this: 9 nights and one reader. Two nights (20 and 21 Nov) were overcast in the evening while GFS
showed almost no cloud, and 13 Nov had GFS cloud of 80-90% over a mostly clear sky; two or three nights decide the
score. GFS cells on Tenerife are coarse (the forecast surface pressure is about 25 hPa below the camera's, so the cell
sits higher than the camera), and the model was fitted on Barcelona airport, not on an island with trade-wind cloud.
The constant guess uses the sample's own mean, which favours it slightly. The Moon (18% to 92% lit) lit clouds from
below on most nights and washed out faint stars; the camera's long exposures show more stars than an eye would.

## Apache Point frames (first test only)

`apo/` holds hourly frames (520 px wide) taken from the Apache Point Observatory all-sky camera's daily timelapse
videos for the nights of 17-18, 18-19 and 19-20 November 2026 (UT), and `apo/labels.json` holds my reading of
each frame (clear / partly / cloudy) with the camera's own UT timestamp read from the overlay. Credit: Apache
Point Observatory (APO / ARC). Used only to check this project's forecast. The reading is a judgement made by
eye, not a measurement. The model's probability for these hours is added by `validate_images.py` once past
weather for the site is available.

## Result on the Apache Point frames (preliminary)

Using real GFS forecasts issued the evening before (7-18 h ahead; `validation/score_apo.py`), on 32 frames over
3 mostly clear nights: Brier 0.127 for the model, 0.111 for the calibrated raw forecast, 0.119 for a constant
guess. The model does not beat either. It missed an overcast spell at 07:47-07:59 UT 18 Nov (forecast cloud 0%).
Too few frames to show skill; the site is far from Barcelona, where the model was fitted.
