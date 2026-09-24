# In-situ HAB data: what exists, and what it can actually validate here

Everything this project measures is self-referential. The forecast is scored against
persistence and against a seasonal climatology, and all three are built from the same
satellite retrieval. None of it answers the question a public-health officer would ask:
**was the bloom real, and was the water dangerous?**

That needs in-situ data. This is a survey of what is public, checked against the live
services on **2026-08-29**, and — more importantly — a measurement of how much of it can
actually be matched to this model's lakes and dates. Reproduce with:

```
python scripts/survey_validation_data.py --report artifacts/validation_survey.json
```

---

## What exists

| Source | Scope | Size | Georeferenced | Access |
|---|---|---|---|---|
| **GLORIA** (PANGAEA) | global, inland + coastal | **7,445 samples, 6,185 with chl-a**, 30 countries, 1990–2022 | yes, 100% | public, CC-BY 4.0 |
| **EPA Water Quality Portal** | United States | **52,270 microcystin measurements**, 4,749 stations, 2015 → 2026 | yes, 100% | public, no account |
| **HAEDAT** (IOC-UNESCO) | global | **14,118 harmful-algae events** since the 1980s | **no** — free text + grid code | public REST API |
| **EEA Waterbase** | Europe | national water-quality reporting | station-level | public |
| **UN GEMStat** | global | national water-quality reporting | station-level | registration |
| **NOAA GLERL** | Lake Erie | operational HAB bulletin — someone else's forecast | yes | public |

Two of these are worth more than the rest:

**GLORIA** is the closest thing to a global validation set for the exact quantity this
model predicts. 6,185 laboratory chlorophyll-a measurements with coordinates and dates,
across 30 countries, of which **2,071 are at ≥20 mg/m³ — genuine bloom conditions** rather
than clear water. Median 8 mg/m³, maximum 13,297 mg/m³.

**The EPA portal carries microcystin**, which is the only dataset here that measures the
thing that actually matters. Chlorophyll is a proxy for algal biomass; microcystin is the
hepatotoxin. **1,473 of the 52,270 measurements exceed the WHO 8 µg/L recreational
guideline.** A forecast validated against this is making a public-health claim rather than
a limnological one.

**HAEDAT is the most complete global record and the least usable here.** 14,118
expert-reviewed events with species, syndrome and toxin detection — but located by free
text ("Ría de Vigo, Ría de Pontevedra…") and internal grid codes, with no coordinates
exposed by the API, and dominated by marine shellfish-toxin closures rather than
freshwater cyanobacteria. Using it would mean geocoding prose.

---

## What can validate *this* model — measured, not assumed

| | inside the 21 modelled lakes | within the ingested window (2016+) |
|---|---|---|
| GLORIA chlorophyll | **1,321** | **422** |
| EPA microcystin | 51 stations | **13 measurements** |

The GLORIA overlap is real and usable: 547 points in western Lake Erie, 264 in the
IJsselmeer, 197 in Taihu, 188 in Kasumigaura. Enough to check whether the satellite
retrieval and the forecast track laboratory chlorophyll on four continents.

**The toxin overlap is almost nothing, and the reason is structural.** 51 stations sit
inside the modelled lakes but produced only 13 measurements since 2016, because the
sampling effort is somewhere else entirely. Of 4,749 microcystin stations, the
best-sampled are small state beach-monitoring sites — Wagon Train Lake, Lake Anita, Nine
Eagles Beach — recreational lakes a few hundred metres across. **Sentinel-3 resolves water
at 300 m and VIIRS at 750 m; a lake needs to be several kilometres across before the
forecaster has twenty pixels to work with.** The toxin-monitoring network and the
satellite-resolvable lakes are close to disjoint populations.

That is worth stating plainly because it cuts against the project's own framing. The app
warns that satellite chlorophyll is not a cyanotoxin measurement and that in-situ sampling
should confirm it — and for most of the lakes where in-situ toxin sampling actually
happens, this model cannot produce a forecast at all.

---

## What was built on it

**The app now plots it.** Clicking any pixel of a forecast draws three things on one
log-scaled chlorophyll axis: the satellite retrieval at that pixel through the input
window, the model's 1-7 day forecast ahead of it, and every public in-situ chlorophyll
measurement within 10 km (25 or 50 on request). Microcystin found nearby is listed beside
the chart rather than drawn on it, because it is a toxin concentration and not
chlorophyll, and a shared axis would be exactly the confusion this project's disclaimer
warns against.

**And the app lists where to look.** A sidebar (`/api/validation-sites`) enumerates the
water bodies that have field data at all: the 21 lakes in the training registry first,
then 150 lakes, bays and seas the archives cover and this model has never seen — Lake
Geneva, Chesapeake Bay, the Gulf of Tonkin, the English Channel. Selecting one frames the
map on it and states what has been measured there and by whom. Water bodies past the
80 km limit for a single forecast get a box at their centre, and are told so.

The list is built by grouping measurements into places, which is where this kind of
catalogue usually goes quietly wrong. Grouping by site name alone collapsed every Silver
Lake in the United States into one "site" 400 km across and every Atlantic sample into an
ocean-sized entry. Points are therefore binned by name *and* degree cell, neighbouring
cells sharing a name are merged, and a name surviving in more than one place is qualified
by its coordinates. Toxin-only sites are left out of this list — several hundred
well-sampled state beach stations that are ponds a few hundred metres across, where a
300 m forecast has nothing to say; their microcystin still reaches the user through the
pixel probe.

    python scripts/build_insitu_index.py      # -> data/validation/insitu.csv.gz

The index is 45,835 rows and 347 kB gzipped, distilled from ~100 MB of downloads:
GLORIA's global chlorophyll, the national microcystin extract, and a per-lake chlorophyll
pull from the Water Quality Portal (a national chlorophyll extract is millions of rows; a
lake-sized bounding box is a hundred, and takes four seconds). It is a local file, so a
click costs a millisecond and reaches no external service.

**The honest part is what happens most of the time.** Measured against a real forecast
over western Lake Erie, anchored 2026-08-26: 165 measurements within 10 km, nearest
930 m - and **zero inside the forecast window**. The 56 that were taken near that point in
the same week of other years are drawn hollow, at reduced weight, labelled "same week
other years", and each carries its true year in its tooltip. They say what an August
normally looks like there. They verify nothing, and the panel says so in those words.

That gap is the finding, not a defect in the plumbing: outside a live monitoring
programme, the in-situ record is years behind the satellite, so a same-window matchup is
the exception. Building the feature is what made the frequency visible.

---

## What would follow from this

Three things, in order of value:

1. **Score the model against GLORIA's 422 in-window matchups.** The app now *shows* the
   comparison one pixel at a time; nothing yet *computes* it over the archive. That is
   the project's first non-self-referential accuracy number and it needs no new data.
2. **Choose lakes to overlap the toxin network.** The registry was assembled from bloom
   literature. Re-selecting the US entries to favour large lakes the EPA portal actually
   samples would convert a 13-measurement overlap into a usable one, and would let the
   model be scored on WHO exceedance rather than on chlorophyll error.
3. **Compare against NOAA's Lake Erie bulletin.** An operational forecast for a lake
   already in the registry, published by the agency that runs the reference system.
   Beating persistence is a low bar; agreeing with GLERL is a real one.

None of these three is done.
