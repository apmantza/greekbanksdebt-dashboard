
# Greek Banks Debt Dashboard

An interactive Streamlit app for exploring Greek banks' debt issuance history and analytics. It loads a curated Excel dataset (`visuals_updated.xlsx`).

The app provides rich visuals and a **Mount Olympus** podiums section.

---

## Features

- **Overview**: Unfiltered issuance data (counts, average size/spread/tenor) and a quarterly sparkline.
- **Overall Issuance**: Cumulative issuance by issuer & type; year-by-year volumes; callable debt by year.
- **Spreads & Coupons**: Scatter visuals vs pricing date and tenor.
- **Liability Profiles**: Upcoming callable debt by year, by issuer and by instrument type.
- **Analytics & Forecasts**:
  - **Average spread by issue type** for future callables.
  - **Expected New Spread** per issue type (trimmed mean of recent deals; SR Preferred=6, Tier2/AT1=4).
  - **Spread Momentum** (Δ bps last 24m vs prior 24m) by issue type and by issuer (Top‑10 by |Δ|).
  - **Tenor–Spread slope (OLS)** for **Senior Preferred** (last 24m, minimum N=3 per issuer).
  - **Issuer Seasonality** by quarter (deal counts; clustered bars + heatmap).
- **Tables**: Filtered data tables and aggregations.
- **Mount Olympus**: Podiums for tightest Senior Preferred, Tier2, AT1; Highest/Lowest spreads; Shortest/Longest tenors. Big medal icons and per‑podium source row preview.
- **Downloads**: Export filtered data to CSV/Excel.
