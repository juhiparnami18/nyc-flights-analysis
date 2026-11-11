# ✈️ Real-World Dataset Analysis: Why Are Flights Delayed at New York's Airports?

> An end-to-end analysis of **336,776 real flights** (JFK, LaGuardia, Newark, 2013): data exploration → cleaning → statistics → visualization → prediction → recommendations.

Built for the **Real-World Dataset Analysis Project** task. The full analysis is in [`analysis.ipynb`](analysis.ipynb) (executed, with all outputs saved).

## 📊 The dataset

| Item | Detail |
|---|---|
| **Flights** | All scheduled departures from JFK, LGA and EWR in 2013 (**336,776 rows**), from the **U.S. Bureau of Transportation Statistics** |
| **Weather** | Hourly airport weather (26,115 rows), originally from **NOAA** |
| **Planes, airlines, airports** | Aircraft registry (age), carrier names and airport names |
| **Access** | `pip install nycflights13` (open data, no manual download) |

## 🎯 Questions answered

How big is the delay problem? Which airlines and airports are worst? When do delays happen? Do delays carry over to arrival? Does weather matter? What drives cancellations? Can delay risk be predicted before departure?

## 🔍 Key findings (all with numbers)

| Finding | Evidence |
|---|---|
| **Scale** | 22.2% of departures and 24.5% of arrivals are 15+ min late; 2.45% of flights are cancelled |
| **Heavy skew** | Mean arrival delay 6.9 min vs median -5 min; worst 10% of late flights = 41% of all delay minutes |
| **Airlines** | Among carriers with 10,000+ flights, delay rate ranges 18.8% (US Airways Inc.) to 32.1% (ExpressJet Airlines Inc.) |
| **Time of day** | 9.4% delayed before 8am vs 33.6% in the evening (5-8pm) |
| **Season** | Jul worst (30.3% delayed), Sep best (14.4%) |
| **Propagation** | Each minute of departure delay adds 1.02 min to arrival delay (R² = 0.84); only 20.1% of late departures arrive on time |
| **Weather** | Precipitation: 40.0% delayed vs 21.0% dry (relative risk 1.91); cancellations 4.1x higher |
| **Cancellations** | 48 days (13% of the year) hold 63% of all cancellations |
| **Prediction** | Test AUC 0.72 (logistic) / 0.78 (gradient boosting); riskiest decile is delayed 58% of the time vs 8% for the safest |

## 🧹 Data cleaning highlights

- **8,255 cancelled** and **458 diverted** flights identified from missing times (kept and analysed separately, never imputed)
- Times stored as `hhmm` integers converted to minutes; `2400` fixed to midnight
- **Impossible wind speed (1,048 mph)** removed; short weather gaps interpolated
- Three validated joins (flights + weather + aircraft + airline) with **no rows lost**

## 🛠️ How to run

```bash
git clone https://github.com/juhiparnami18/nyc-flights-analysis.git
cd nyc-flights-analysis
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab                        # open analysis.ipynb -> Run All
python -m unittest discover tests -v
```

## 🗂️ Structure

```
nyc-flights-analysis/
├── analysis.ipynb        # The full analysis (executed)
├── requirements.txt
├── src/cleaning.py       # Tested helper functions
├── tests/test_cleaning.py
└── outputs/
    ├── figures/          # 15 charts
    ├── tables/           # CSV summaries
    └── results.json      # Every key number used in the report
```

## ⚠️ Limitations (summary; full table in notebook section 8)

One year and three airports only; 2013 data is dated; cancelled/diverted flights have no delay (survivorship bias); no official delay-cause codes; observations are not independent (delays cascade through aircraft and crews); correlation is not causation; the model is a moderate-accuracy screening tool.

---
**Author:** Juhi Parnami · B.Com student & aspiring Data Scientist
