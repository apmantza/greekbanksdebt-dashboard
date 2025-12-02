# Greek Banks Debt Dashboard

## Overview
This Streamlit app provides an interactive dashboard to analyze Greek banks' debt issuances. It includes:

### Key Features
- **Filters**: Filter data by Issuer, Issue Type, ESG Label, Pricing Date range, and Maturity.
- **Summary Metrics**:
  - Total Issuances
  - Cumulative Issuance Size
  - Cumulative Issuance by Issuer and Issue Type
  - Average Issuance per Year
  - Debt Maturing Next Year
  - Average Spread of Debt Maturing Next Year
- **Visualizations**:
  - Per Bank Trend: Average Spread per Year by Bank
  - Scatter Plot: Debt Issuances over time
  - Liability Profiles:
    - Issuance Size per Year by Bank (current and future years only)
    - Issuance Size per Year by Issue Type (current and future years only)
  - Debt Maturing Next Year by Issuer
- **Download Options**:
  - Filtered data in CSV and Excel formats
  - Charts in HTML format (compatible without Kaleido)

### Requirements
The app requires the following Python packages:
```
streamlit>=1.30.0
pandas>=2.0.0
plotly>=5.18.0
openpyxl>=3.1.0
```

### Deployment Instructions
1. Place the following files in your GitHub repository:
   - `app_final_downloads_corrected.py` (main app file)
   - `requirements.txt` (dependencies)
   - `visuals_updated.xlsx` (data file)

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app:
   - Connect your GitHub account.
   - Select the repository containing the files.
   - Set **Main file path** to `app_final_downloads_corrected.py`.

3. Deploy the app. Streamlit Cloud will install dependencies from `requirements.txt`.

### Notes
- Liability profiles dynamically filter out past years and only show current and future years.
- All chart downloads are provided in HTML format for maximum compatibility.

