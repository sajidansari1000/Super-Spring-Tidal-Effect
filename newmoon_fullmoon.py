import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from skyfield.api import load, wgs84
from skyfield import almanac

# ==========================
# CONFIGURATION
# ==========================
EARTHQUAKE_FILE = "usgs_earthquakes.json"
EPHEMERIS_FILE  = "de433.bsp"

START_DATE = datetime(1970, 1, 1, tzinfo=timezone.utc)
END_DATE   = datetime(2025, 12, 31, tzinfo=timezone.utc)

NEW_MOON_WINDOW_HOURS = 36     # ± hours
PERIGEE_WINDOW_DAYS  = 2       # ± days
PERIGEE_STEP_HOURS   = 6       # sampling resolution
DELTA_LAT_THRESHOLD  = 10       # degrees for "tight" alignment

# ==========================
# LOAD EARTHQUAKES
# ==========================
print("Loading earthquakes...")
with open(EARTHQUAKE_FILE, "r", encoding="utf-8") as f:
    eq_raw = json.load(f)

earthquakes = []
for feature in eq_raw["features"]:
    try:
        t_ms = feature["properties"]["time"]
        if t_ms is None:
            continue
        qt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
        if not (START_DATE <= qt <= END_DATE):
            continue
        earthquakes.append({
            "quake_time": qt,
            "magnitude": feature["properties"]["mag"],
            "place": feature["properties"]["place"]
        })
    except Exception:
        continue

eq_df = pd.DataFrame(earthquakes)
print(f"Earthquakes loaded: {len(eq_df)}")

# ==========================
# LOAD EPHEMERIS
# ==========================
print("Loading ephemeris (DE433)...")
ts  = load.timescale()
eph = load(EPHEMERIS_FILE)

earth = eph["earth"]
moon  = eph["moon"]
sun   = eph["sun"]

# ==========================
# NEW & FULL MOONS
# ==========================
print("Computing New & Full Moons...")
t0 = ts.from_datetime(START_DATE)
t1 = ts.from_datetime(END_DATE)

times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))

new_moons  = []
full_moons = []

for t, p in zip(times, phases):
    dt = t.utc_datetime().replace(tzinfo=timezone.utc)
    if p == 0:      # New Moon
        new_moons.append(dt)
    elif p == 2:    # Full Moon
        full_moons.append(dt)

print(f"New Moons found: {len(new_moons)}")
print(f"Full Moons found: {len(full_moons)}")

# ==========================
# LUNAR PERIGEES
# ==========================
print("Computing lunar perigees (DE433)...")
perigees = []
t = START_DATE
prev_dist = None
prev_time = None

while t <= END_DATE:
    tt = ts.from_datetime(t)
    dist = earth.at(tt).observe(moon).distance().km
    if prev_dist is not None and dist > prev_dist:
        perigees.append(prev_time)
    prev_dist = dist
    prev_time = t
    t += timedelta(hours=PERIGEE_STEP_HOURS)

print(f"Perigees found: {len(perigees)}")

# ==========================
# HELPER FUNCTIONS
# ==========================
def nearest_time(target, time_list):
    deltas = [abs((target - t).total_seconds()) for t in time_list]
    idx = np.argmin(deltas)
    return time_list[idx], deltas[idx]

def subpoint(body, dt):
    t = ts.from_datetime(dt)
    apparent = earth.at(t).observe(body).apparent()
    sp = wgs84.subpoint(apparent)
    return sp.latitude.degrees, sp.longitude.degrees

# ==========================
# TIDAL MATCH FUNCTION
# ==========================
def find_tidal_matches(eq_df, phase_times, phase_name):
    results = []

    for _, eq in eq_df.iterrows():
        qt = eq["quake_time"]

        phase_time, phase_sec = nearest_time(qt, phase_times)
        if phase_sec > NEW_MOON_WINDOW_HOURS * 3600:
            continue

        pg_time, pg_sec = nearest_time(qt, perigees)
        if pg_sec > PERIGEE_WINDOW_DAYS * 86400:
            continue

        sublunar_lat, sublunar_lon = subpoint(moon, pg_time)
        subsolar_lat, subsolar_lon   = subpoint(sun,  pg_time)

        delta_lat = abs(sublunar_lat - subsolar_lat)

        results.append({
            "phase": phase_name,
            "quake_time": qt,
            "magnitude": eq["magnitude"],
            "place": eq["place"],

            "phase_time": phase_time,
            "phase_offset_hours": phase_sec / 3600,

            "perigee_time": pg_time,
            "perigee_offset_days": pg_sec / 86400,

            "sublunar_lat": sublunar_lat,
            "sublunar_lon": sublunar_lon,
            "subsolar_lat": subsolar_lat,
            "subsolar_lon": subsolar_lon,

            "delta_lat": delta_lat
        })

    return pd.DataFrame(results)

# ==========================
# RUN MATCHING
# ==========================
print("Matching New Moon cases...")
df_new = find_tidal_matches(eq_df, new_moons, "New Moon")

print("Matching Full Moon cases...")
df_full = find_tidal_matches(eq_df, full_moons, "Full Moon")

# ==========================
# FILTER TIGHT ALIGNMENTS
# ==========================
df_new_tight = df_new[df_new["delta_lat"] <= DELTA_LAT_THRESHOLD]
df_full_tight = df_full[df_full["delta_lat"] <= DELTA_LAT_THRESHOLD]

# ==========================
# SAVE RESULTS
# ==========================
df_new.to_csv("new_moon_perigee_earthquakes.csv", index=False)
df_full.to_csv("full_moon_perigee_earthquakes.csv", index=False)
df_new_tight.to_csv("tight_new_moon_perigee_earthquakes.csv", index=False)
df_full_tight.to_csv("tight_full_moon_perigee_earthquakes.csv", index=False)

# ==========================
# PRINT SUMMARIES
# ==========================
print("\nNew Moon Matches:", len(df_new))
print("New Moon Tight Matches (Δlat ≤ 10°):", len(df_new_tight))

print("\nFull Moon Matches:", len(df_full))
print("Full Moon Tight Matches (Δlat ≤ 10°):", len(df_full_tight))
