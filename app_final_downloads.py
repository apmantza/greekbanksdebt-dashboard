
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import datetime
import re

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Greek Banks Debt Dashboard", layout="wide")

# -----------------------------------------------------------------------------
# Theme toggle: Mount Olympus ON by default; toggle to Classic
# -----------------------------------------------------------------------------
classic_theme = st.sidebar.checkbox("Switch to Classic Theme", value=False)
flamboyant = not classic_theme

# Header
if flamboyant:
    st.markdown("## 🗻 Greek Banks Debt Dashboard ✨", unsafe_allow_html=True)
else:
    st.title("Greek Banks Debt Dashboard")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
FILE_PATH = 'visuals_updated.xlsx'
df = pd.read_excel(FILE_PATH, sheet_name='Sheet1', engine='openpyxl')
df.columns = [c.strip() for c in df.columns]

# Make intent explicit: 'Structure' = text (e.g., 6NC5/PNC5), 'Maturity Date' = final date
rename_map = {}
if 'Maturity' in df.columns:
    rename_map['Maturity'] = 'Structure'
if 'Maturity.1' in df.columns:
    rename_map['Maturity.1'] = 'Maturity Date'
df = df.rename(columns=rename_map)

# -----------------------------------------------------------------------------
# Datetime conversions (single pass)
# -----------------------------------------------------------------------------
for col in ['Pricing Date', 'FIRST_CALL', 'Maturity Date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# -----------------------------------------------------------------------------
# Original Tenor = number AFTER "NC" (per your clarification; works for PERPs too)
# Examples: 6NC5 -> 5 ; 3,5NC2,5 -> 2.5 ; PNC5 -> 5
# -----------------------------------------------------------------------------
def extract_tenor(structure: str):
    if isinstance(structure, str) and 'NC' in structure.upper():
        m = re.search(r'NC\s*(\d+(?:[.,]\d+)?)', structure, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', '.'))
    return np.nan

df['Original Tenor'] = df['Structure'].apply(extract_tenor)

# -----------------------------------------------------------------------------
# Numeric conversions
# -----------------------------------------------------------------------------
for col in ['Size', 'Coupon', 'Re-offer Spread']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# --- NORMALIZE ISSUE TYPE (spaces/case) -------------------------------------
# Ensures exact matches: 'SR Preferred', 'Tier2', 'AT1'
df['Issue Type Clean'] = df['Issue Type'].astype(str).str.strip()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def first_call_year_bounds(df_in: pd.DataFrame):
    """Return (min_year, max_year, default_T_plus_1_clamped) from FIRST_CALL."""
    if 'FIRST_CALL' not in df_in.columns:
        y = datetime.now().year
        return y, y, y
    years = df_in['FIRST_CALL'].dt.year.dropna()
    current_year = datetime.now().year
    if years.empty:
        return current_year, current_year, current_year
    min_y, max_y = int(years.min()), int(years.max())
    t_plus_1 = current_year + 1
    default_clamped = max(min_y, min(t_plus_1, max_y))
    return min_y, max_y, default_clamped

FC_MIN, FC_MAX, FC_DEFAULT = first_call_year_bounds(df)
UNFILTERED_MARK = " †"

def safe_idxmin(s: pd.Series):
    s = s.dropna()
    return s.idxmin() if not s.empty else None

def safe_idxmax(s: pd.Series):
    s = s.dropna()
    return s.idxmax() if not s.empty else None

def add_quarter_fields(df_in: pd.DataFrame, date_col='Pricing Date'):
    out = df_in.dropna(subset=[date_col]).copy()
    out[date_col] = pd.to_datetime(out[date_col], dayfirst=True, errors='coerce')
    out = out.dropna(subset=[date_col])
    q_period = out[date_col].dt.to_period('Q')
    out['Quarter Start'] = q_period.dt.start_time
    out['Quarter Label'] = "Q" + q_period.dt.quarter.astype(str) + " " + out[date_col].dt.year.astype(str)
    return out

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header('Filters')

issuers = st.sidebar.multiselect(
    'Issuer(s):',
    sorted(df['Issuer'].dropna().unique()),
    default=sorted(df['Issuer'].dropna().unique())
)

issue_types = st.sidebar.multiselect(
    'Issue Type(s):',
    sorted(df['Issue Type'].dropna().unique()),
    default=sorted(df['Issue Type'].dropna().unique())
)

esg_labels = st.sidebar.multiselect(
    'ESG Label(s):',
    sorted(df['ESG Label'].dropna().unique()),
    default=sorted(df['ESG Label'].dropna().unique())
)

valid_dates = df['Pricing Date'].dropna()
if not valid_dates.empty:
    min_date, max_date = valid_dates.min().to_pydatetime(), valid_dates.max().to_pydatetime()
    date_range = st.sidebar.slider('Pricing Date Range:', min_value=min_date, max_value=max_date, value=(min_date, max_date))
else:
    date_range = (None, None)

call_valid_dates = df['FIRST_CALL'].dropna()
if not call_valid_dates.empty:
    call_min, call_max = call_valid_dates.min().to_pydatetime(), call_valid_dates.max().to_pydatetime()
    call_range = st.sidebar.slider('Call Date Range:', min_value=call_min, max_value=call_max, value=(call_min, call_max))
else:
    call_range = (None, None)

tenor_series = df['Original Tenor'].dropna()
tenor_min, tenor_max = (float(tenor_series.min()), float(tenor_series.max())) if not tenor_series.empty else (0.0, 0.0)
tenor_range = st.sidebar.slider('Original Tenor (Years):', min_value=tenor_min, max_value=tenor_max, value=(tenor_min, tenor_max), step=0.25)

# -----------------------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------------------
filtered_df = df.copy()
if issuers:
    filtered_df = filtered_df[filtered_df['Issuer'].isin(issuers)]
if issue_types:
    filtered_df = filtered_df[filtered_df['Issue Type'].isin(issue_types)]
if esg_labels:
    filtered_df = filtered_df[filtered_df['ESG Label'].isin(esg_labels)]
if date_range[0] and date_range[1]:
    filtered_df = filtered_df[
        (filtered_df['Pricing Date'] >= date_range[0]) &
        (filtered_df['Pricing Date'] <= date_range[1])
    ]
if call_range[0] and call_range[1]:
    filtered_df = filtered_df[
        (filtered_df['FIRST_CALL'] >= call_range[0]) &
        (filtered_df['FIRST_CALL'] <= call_range[1])
    ]
filtered_df = filtered_df[
    (filtered_df['Original Tenor'] >= tenor_range[0]) &
    (filtered_df['Original Tenor'] <= tenor_range[1])
]

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
overview_tab, issuance_tab, spreads_tab, liability_tab, analytics_tab, tables_tab, olympus_tab, downloads_tab = st.tabs([
    'Overview', 'Overall Issuance', 'Spreads & Coupons', 'Liability Profiles', 'Analytics & Forecasts', 'Tables', 'Mount Olympus', 'Downloads'
])

# -----------------------------------------------------------------------------
# Overview Tab (uses df, not filtered_df)
# -----------------------------------------------------------------------------
with overview_tab:
    if flamboyant:
        st.markdown("### ✨ Overview ✨", unsafe_allow_html=True)
    st.write("Use filters on the left to customize other tabs. Overview shows unfiltered KPIs.")

    # Helper formatters
    format_bps = lambda x: f"{x:.1f} bps" if pd.notna(x) else "N/A"
    format_million = lambda x: f"{x:.1f}M" if pd.notna(x) else "N/A"  # 'Size' values are in millions
    format_years = lambda x: f"{x:.1f} yrs" if pd.notna(x) else "N/A"

    # UNFILTERED subsets from full dataset
    seniors_full = df[df['Issue Type'].str.contains('SR Preferred', case=False, na=False)]
    tier2_full = df[df['Issue Type'].str.contains('Tier2', case=False, na=False)]
    at1_full = df[df['Issue Type'].str.contains('AT1', case=False, na=False)]

    # 1st line: ALL ISSUANCES (UNFILTERED)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.metric('🧮 Total Issuances', f"{len(df):,}")
    with r1c2:
        st.metric('💸 Average Issuance Size', format_million(df['Size'].mean()))
    with r1c3:
        st.metric('📉 Average Issuance Spread', format_bps(df['Re-offer Spread'].mean()))
    with r1c4:
        st.metric('⏱️ Average Tenor', format_years(df['Original Tenor'].mean()))

    # 2nd line: SENIORS — UNFILTERED
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.metric('🏦 Seniors – Count', f"{len(seniors_full):,}")
    with r2c2:
        st.metric('💸 Seniors – Avg Size', format_million(seniors_full['Size'].mean()))
    with r2c3:
        st.metric('📉 Seniors – Avg Spread', format_bps(seniors_full['Re-offer Spread'].mean()))
    with r2c4:
        st.metric('⏱️ Seniors – Avg Tenor', format_years(seniors_full['Original Tenor'].mean()))

    # 3rd line: TIER2 — UNFILTERED
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        st.metric('🏦 Tier2 – Count', f"{len(tier2_full):,}")
    with r3c2:
        st.metric('💸 Tier2 – Avg Size', format_million(tier2_full['Size'].mean()))
    with r3c3:
        st.metric('📉 Tier2 – Avg Spread', format_bps(tier2_full['Re-offer Spread'].mean()))
    with r3c4:
        st.metric('⏱️ Tier2 – Avg Tenor', format_years(tier2_full['Original Tenor'].mean()))

    # 4th line: AT1 — UNFILTERED
    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    with r4c1:
        st.metric('🏦 AT1 – Count', f"{len(at1_full):,}")
    with r4c2:
        st.metric('💸 AT1 – Avg Size', format_million(at1_full['Size'].mean()))
    with r4c3:
        st.metric('📉 AT1 – Avg Spread', format_bps(at1_full['Re-offer Spread'].mean()))
    with r4c4:
        st.metric('⏱️ AT1 – Avg Tenor', format_years(at1_full['Original Tenor'].mean()))

    # Year-on-Year Deltas (unfiltered)
    st.markdown("#### 📈 Year-on-Year Deltas")
    df_y = df.dropna(subset=['Pricing Date']).copy()
    df_y['Pricing Date'] = pd.to_datetime(df_y['Pricing Date'], dayfirst=True, errors='coerce')
    df_y = df_y.dropna(subset=['Pricing Date'])
    df_y['Year'] = df_y['Pricing Date'].dt.year

    if df_y['Year'].nunique() >= 2:
        last_year = int(df_y['Year'].max())
        prev_year = last_year - 1
        last_data = df_y[df_y['Year'] == last_year]
        prev_data = df_y[df_y['Year'] == prev_year]

        def pct_change(prev, last):
            try:
                if prev is None or np.isnan(prev) or prev == 0:
                    return np.nan
                return (last - prev) / prev * 100.0
            except Exception:
                return np.nan

        def fmt_abs(x, unit=""):
            return f"{x:.1f}{unit}" if pd.notna(x) else "N/A"

        def fmt_pct(p):
            return f"{p:+.1f}%" if pd.notna(p) else "N/A"

        # 1) Count
        count_last, count_prev = len(last_data), len(prev_data)
        delta_count = count_last - count_prev
        pct_count = pct_change(count_prev, count_last)
        delta_color_count = "normal" if delta_count >= 0 else "inverse"

        # 2) Total Size (M)
        size_last = last_data['Size'].sum()
        size_prev = prev_data['Size'].sum()
        delta_size = size_last - size_prev
        pct_size = pct_change(size_prev, size_last)
        delta_color_size = "normal" if delta_size >= 0 else "inverse"

        # 3) Average Spread (bps) — widening is 'inverse'
        spread_last = last_data['Re-offer Spread'].mean()
        spread_prev = prev_data['Re-offer Spread'].mean()
        delta_spread = spread_last - spread_prev
        pct_spread = pct_change(spread_prev, spread_last)
        delta_color_spread = "inverse" if delta_spread >= 0 else "normal"

        # 4) Average Tenor (yrs)
        tenor_last = last_data['Original Tenor'].mean()
        tenor_prev = prev_data['Original Tenor'].mean()
        delta_tenor = tenor_last - tenor_prev
        pct_tenor = pct_change(tenor_prev, tenor_last)
        delta_color_tenor = "off"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                f"Issuances {last_year}",
                f"{count_last:,}",
                delta=f"{delta_count:+,} ({fmt_pct(pct_count)})",
                delta_color=delta_color_count
            )
        with k2:
            st.metric(
                f"Total Size {last_year}",
                f"{size_last:.0f}M",
                delta=f"{delta_size:+.0f}M ({fmt_pct(pct_size)})",
                delta_color=delta_color_size
            )
        with k3:
            st.metric(
                f"Avg Spread {last_year}",
                f"{fmt_abs(spread_last, ' bps')}",
                delta=f"{fmt_abs(delta_spread, ' bps')} ({fmt_pct(pct_spread)})",
                delta_color=delta_color_spread
            )
        with k4:
            st.metric(
                f"Avg Tenor {last_year}",
                f"{fmt_abs(tenor_last, ' yrs')}",
                delta=f"{fmt_abs(delta_tenor, ' yrs')} ({fmt_pct(pct_tenor)})",
                delta_color=delta_color_tenor
            )

    # Issuance Sparkline (Quarterly, unfiltered)
    st.markdown("#### 📊 Issuance Sparkline (Quarterly)")
    qdf = add_quarter_fields(df, 'Pricing Date')
    if not qdf.empty:
        data_min_year = int(qdf['Pricing Date'].dt.year.min())
        data_max_year = int(qdf['Pricing Date'].dt.year.max())
        default_start = max(2019, data_min_year)
        year_start, year_end = st.slider(
            "Select year range for sparkline",
            min_value=data_min_year,
            max_value=data_max_year,
            value=(default_start, data_max_year),
            step=1,
            help="Adjust the year range to focus the quarterly sparkline."
        )
        qdf = qdf[(qdf['Pricing Date'].dt.year >= year_start) & (qdf['Pricing Date'].dt.year <= year_end)]
        spark = (
            qdf.groupby(['Quarter Start', 'Quarter Label'])['Size']
            .sum()
            .reset_index()
            .sort_values('Quarter Start')
        )
        if not spark.empty:
            fig_spark = px.area(spark, x='Quarter Label', y='Size', title=None)
            fig_spark.update_layout(
                height=180,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title='',
                yaxis_title='',
                xaxis=dict(categoryorder='array', categoryarray=spark['Quarter Label'].tolist(), showgrid=False),
            )
            fig_spark.update_traces(
                line=dict(color='#2c3e50'),
                fillcolor='rgba(44, 62, 80, 0.20)',
                hovertemplate="Quarter=%{x}<br>Size=%{y:.0f}M<extra></extra>"
            )
            st.plotly_chart(fig_spark, use_container_width=True)
        else:
            st.caption("No quarterly issuance found for the selected year range.")
    else:
        st.caption("No pricing dates available in the full dataset to produce the sparkline.")

# -----------------------------------------------------------------------------
# Overall Issuance Tab
# -----------------------------------------------------------------------------
with issuance_tab:
    if flamboyant:
        st.markdown("### 📈 Overall Issuance", unsafe_allow_html=True)

    # 1) Cumulative Issuance by Issuer & Type
    if {'Issuer', 'Issue Type', 'Size'}.issubset(filtered_df.columns):
        cum_df = filtered_df.groupby(['Issuer', 'Issue Type'])['Size'].sum().reset_index()
        fig_cum = px.bar(cum_df, x='Issuer', y='Size', color='Issue Type', barmode='group',
                         title='Cumulative Issuance by Issuer and Issue Type')
        fig_cum.update_layout(yaxis_title='Size (M)', xaxis_title='Issuer', yaxis_tickformat=',.0f')
        fig_cum.update_traces(hovertemplate="Issuer=%{x}<br>Size=%{y:.0f}M<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig_cum, use_container_width=True)

    # 2) Issuance Size by Year
    if {'Year issued', 'Size'}.issubset(filtered_df.columns):
        year_df = filtered_df.groupby('Year issued')['Size'].sum().reset_index()
        fig_year = px.bar(year_df, x='Year issued', y='Size', title='Issuance Size by Year')
        fig_year.update_layout(yaxis_title='Size (M)', xaxis_title='Year', yaxis_tickformat=',.0f')
        fig_year.update_traces(hovertemplate="Year=%{x}<br>Size=%{y:.0f}M<extra></extra>")
        st.plotly_chart(fig_year, use_container_width=True)

    # 3) Callable debt per selected year (T+1 default, centralized bounds)
    if 'FIRST_CALL' in filtered_df.columns and not filtered_df['FIRST_CALL'].dropna().empty:
        fixed_year = st.slider('Select Year for Callable Debt:',
                               min_value=FC_MIN, max_value=FC_MAX, value=FC_DEFAULT, step=1)
        st.caption(
            f"ℹ️ Default set to next calendar year (T+1 = {datetime.now().year + 1}). "
            f"If unavailable in the data, it falls back within [{FC_MIN} … {FC_MAX}]."
        )
        calls_fixed = filtered_df[filtered_df['FIRST_CALL'].dt.year == fixed_year]
        if not calls_fixed.empty:
            debt_fixed = calls_fixed.groupby(['Issuer', 'Issue Type'])['Size'].sum().reset_index()
            fig_fixed = px.bar(debt_fixed, x='Issuer', y='Size', color='Issue Type', barmode='group',
                               title=f'Debt Callables in {fixed_year}')
            fig_fixed.update_layout(yaxis_title='Size (M)', xaxis_title='Issuer', yaxis_tickformat=',.0f')
            fig_fixed.update_traces(hovertemplate="Issuer=%{x}<br>Size=%{y:.0f}M<extra>%{legendgroup}</extra>")
            st.plotly_chart(fig_fixed, use_container_width=True)
        else:
            st.info(f"No callable debt found for {fixed_year}.")
    else:
        st.info("No FIRST_CALL data available to configure the callable debt slider.")

    # 4) Issuer share of total issuance (pie)
    share_df = filtered_df.groupby('Issuer')['Size'].sum().reset_index()
    if not share_df.empty:
        fig_share = px.pie(share_df, names='Issuer', values='Size', title='Issuer Share of Total Issuance')
        st.plotly_chart(fig_share, use_container_width=True)

    # 5) Share of Issue Types (pie)
    type_share = filtered_df.groupby('Issue Type')['Size'].sum().reset_index()
    if not type_share.empty:
        fig_types = px.pie(type_share, names='Issue Type', values='Size', title='Share of Issue Types')
        st.plotly_chart(fig_types, use_container_width=True)

# -----------------------------------------------------------------------------
# Spreads & Coupons Tab
# -----------------------------------------------------------------------------
with spreads_tab:
    if flamboyant:
        st.markdown("### 🎨 Spreads & Coupons", unsafe_allow_html=True)

    if {'Pricing Date', 'Re-offer Spread', 'Issuer'}.issubset(filtered_df.columns):
        fig1 = px.scatter(filtered_df, x='Pricing Date', y='Re-offer Spread', color='Issuer',
                          title="Spread vs Pricing Date")
        fig1.update_layout(yaxis_title='Re-offer Spread (bps)', xaxis_title='Pricing Date', yaxis_tickformat=',.1f')
        fig1.update_traces(hovertemplate="Date=%{x|%d/%m/%Y}<br>Spread=%{y:.1f} bps<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig1, use_container_width=True)

    if {'Pricing Date', 'Coupon', 'Issuer'}.issubset(filtered_df.columns):
        fig2 = px.scatter(filtered_df, x='Pricing Date', y='Coupon', color='Issuer',
                          title="Coupon vs Pricing Date")
        fig2.update_layout(yaxis_title='Coupon (%)', xaxis_title='Pricing Date', yaxis_tickformat=',.2f')
        fig2.update_traces(hovertemplate="Date=%{x|%d/%m/%Y}<br>Coupon=%{y:.2f}%<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig2, use_container_width=True)

    if {'Original Tenor', 'Re-offer Spread', 'Issuer'}.issubset(filtered_df.columns):
        fig3 = px.scatter(
            filtered_df.dropna(subset=['Original Tenor', 'Re-offer Spread']),
            x='Original Tenor', y='Re-offer Spread', color='Issuer', trendline='lowess',
            title='Re-offer Spread vs Original Tenor'
        )
        fig3.update_layout(xaxis_title='Tenor (yrs)', yaxis_title='Re-offer Spread (bps)', yaxis_tickformat=',.1f')
        fig3.update_traces(hovertemplate="Tenor=%{x:.1f} yrs<br>Spread=%{y:.1f} bps<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig3, use_container_width=True)

    if {'Original Tenor', 'Coupon', 'Issuer'}.issubset(filtered_df.columns):
        fig4 = px.scatter(
            filtered_df.dropna(subset=['Original Tenor', 'Coupon']),
            x='Original Tenor', y='Coupon', color='Issuer', trendline='lowess',
            title='Coupon vs Original Tenor'
        )
        fig4.update_layout(xaxis_title='Tenor (yrs)', yaxis_title='Coupon (%)', yaxis_tickformat=',.2f')
        fig4.update_traces(hovertemplate="Tenor=%{x:.1f} yrs<br>Coupon=%{y:.2f}%<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------------------------------------------
# Liability Profiles Tab
# -----------------------------------------------------------------------------
with liability_tab:
    if flamboyant:
        st.markdown("### 🏦 Liability Profiles", unsafe_allow_html=True)

    call_df = filtered_df.dropna(subset=['FIRST_CALL']).copy()
    call_df['Call Year'] = call_df['FIRST_CALL'].dt.year
    call_df = call_df[call_df['Call Year'] >= datetime.now().year]

    if not call_df.empty:
        liab_bank = call_df.groupby(['Call Year', 'Issuer'])['Size'].sum().reset_index()
        fig_liab_bank = px.bar(liab_bank, x='Call Year', y='Size', color='Issuer', barmode='group',
                               title='Liability Profile by Bank')
        fig_liab_bank.update_layout(yaxis_title='Size (M)', xaxis_title='Call Year', yaxis_tickformat=',.0f')
        fig_liab_bank.update_traces(hovertemplate="Year=%{x}<br>Size=%{y:.0f}M<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig_liab_bank, use_container_width=True)

        liab_type = call_df.groupby(['Call Year', 'Issue Type'])['Size'].sum().reset_index()
        fig_liab_type = px.bar(liab_type, x='Call Year', y='Size', color='Issue Type', barmode='group',
                               title='Liability Profile by Issue Type')
        fig_liab_type.update_layout(yaxis_title='Size (M)', xaxis_title='Call Year', yaxis_tickformat=',.0f')
        fig_liab_type.update_traces(hovertemplate="Year=%{x}<br>Size=%{y:.0f}M<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig_liab_type, use_container_width=True)
    else:
        st.info("No upcoming callable debt found.")




# -----------------------------------------------------------------------------
# Analytics & Forecasts Tab  (UNFILTERED DATA)
# -----------------------------------------------------------------------------
with analytics_tab:
    # NOTE: This tab uses the full dataset (unfiltered), regardless of filters.
    st.info("Note: This tab uses the full dataset (unfiltered) for analytics and forecasts.")
    base_df = df.copy()  # always unfiltered here

    # Ensure dates are parsed
    if 'Pricing Date' in base_df.columns:
        base_df['Pricing Date'] = pd.to_datetime(base_df['Pricing Date'], dayfirst=True, errors='coerce')
    if 'FIRST_CALL' in base_df.columns:
        base_df['FIRST_CALL'] = pd.to_datetime(base_df['FIRST_CALL'], dayfirst=True, errors='coerce')

    # Normalize Issue Type once for robust exact matching
    base_df['Issue Type Clean'] = base_df['Issue Type'].astype(str).str.strip()

    # ===========================================
    # A) EXPECTED NEW SPREAD (Trimmed mean, type-specific windows)
    # ===========================================
    st.markdown("### 🎯 Expected New Spread per Issue Type (Trimmed Mean on Recent Deals)")

    # Rules:
    # - SR Preferred: last 6 deals -> drop min & max -> average remaining 4
    # - Tier2 & AT1:  last 4 deals -> drop min & max -> average remaining 2
    # - Skip a type if it doesn't meet the minimum window length
    expected_rows = []
    for issue_type in ['SR Preferred', 'Tier2', 'AT1']:
        d = (base_df[base_df['Issue Type Clean'] == issue_type]
             .dropna(subset=['Re-offer Spread', 'Pricing Date'])
             .sort_values('Pricing Date'))

        N = 6 if issue_type == 'SR Preferred' else 4
        values = d.tail(N)['Re-offer Spread'].tolist()

        if len(values) >= N and N >= 4:
            vals = values.copy()
            vals.remove(max(vals))
            vals.remove(min(vals))
            expected = sum(vals) / len(vals) if len(vals) > 0 else np.nan
            if pd.notna(expected):
                expected_rows.append({'Issue Type': issue_type, 'Expected New Spread (bps)': expected})

    if expected_rows:
        exp_df = (pd.DataFrame(expected_rows)
                  .dropna(subset=['Expected New Spread (bps)'])
                  .sort_values('Expected New Spread (bps)'))
        fig_exp = px.bar(
            exp_df, x='Issue Type', y='Expected New Spread (bps)',
            title='Expected New Spread (Trimmed mean on recent deals; SRP=6, Tier2/AT1=4)',
            text=exp_df['Expected New Spread (bps)'].map(lambda v: f"{v:.1f} bps")
        )
        fig_exp.update_layout(yaxis_title='Expected New Spread (bps)', xaxis_title='Issue Type', yaxis_tickformat=',.1f')
        fig_exp.update_traces(textposition='outside')
        st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.info("Insufficient recent deals to compute trimmed means by issue type.")

    # ===========================================================
    # B) Average spread per Issue Type (Future callables only)
    # ===========================================================
    st.markdown("### ⏳ Spreads by Issue Type — Future Callables Only")
    today = pd.Timestamp.today().normalize()
    fc_future = base_df.dropna(subset=['Re-offer Spread', 'FIRST_CALL']).copy()
    fc_future = fc_future[fc_future['FIRST_CALL'] >= today]

    avg_future = (
        fc_future.groupby('Issue Type')['Re-offer Spread']
                 .mean()
                 .reset_index()
                 .sort_values('Re-offer Spread')
    )
    if not avg_future.empty:
        fig_avg_future = px.bar(
            avg_future, x='Issue Type', y='Re-offer Spread',
            title='Average Re-offer Spread by Issue Type — Future Callables (FIRST_CALL in the future)',
            text=avg_future['Re-offer Spread'].map(lambda v: f"{v:.1f} bps")
        )
        fig_avg_future.update_layout(yaxis_title='Re-offer Spread (bps)', xaxis_title='Issue Type', yaxis_tickformat=',.1f')
        fig_avg_future.update_traces(textposition='outside')
        st.plotly_chart(fig_avg_future, use_container_width=True)
    else:
        st.info("No instruments with FIRST_CALL in the future were found to compute this metric.")

    # ======================================
    # C) SPREAD MOMENTUM (Δ 24m vs prior 24m) — min count per window = 2
    # ======================================
    st.markdown("### 📈 Spread Momentum (Δ bps) — Last 24 Months vs Prior 24 Months")

    def compute_spread_momentum(df_in: pd.DataFrame, group_col: str, min_count_per_window: int = 2):
        d = df_in.dropna(subset=['Re-offer Spread', 'Pricing Date']).copy()
        now = pd.Timestamp.today().normalize()
        last_start = now - pd.DateOffset(months=24)
        prior_start = now - pd.DateOffset(months=48)

        last = d[(d['Pricing Date'] >= last_start) & (d['Pricing Date'] <= now)]
        prior = d[(d['Pricing Date'] >= prior_start) & (d['Pricing Date'] < last_start)]

        # counts & means per window
        g_last_cnt = last.groupby(group_col)['Re-offer Spread'].count().rename('Count Last 24m')
        g_prior_cnt = prior.groupby(group_col)['Re-offer Spread'].count().rename('Count Prior 24m')
        g_last = last.groupby(group_col)['Re-offer Spread'].mean().rename('Last 24m (bps)')
        g_prior = prior.groupby(group_col)['Re-offer Spread'].mean().rename('Prior 24m (bps)')

        joined = pd.concat([g_last, g_last_cnt, g_prior, g_prior_cnt], axis=1).dropna()

        # gate by counts per window (>= 2 each)
        joined = joined[(joined['Count Last 24m'] >= min_count_per_window) &
                        (joined['Count Prior 24m'] >= min_count_per_window)]
        if joined.empty:
            return pd.DataFrame(), last_start, prior_start, now

        joined['Δ (bps)'] = joined['Last 24m (bps)'] - joined['Prior 24m (bps)']
        joined = joined.reset_index()
        return joined, last_start, prior_start, now

    # Momentum by Issue Type (all)
    mom_issue, last_start, prior_start, now_ts = compute_spread_momentum(base_df, 'Issue Type', min_count_per_window=2)
    if not mom_issue.empty:
        mom_issue_sorted = mom_issue.sort_values('Δ (bps)')
        fig_mom_issue = px.bar(
            mom_issue_sorted, x='Issue Type', y='Δ (bps)',
            color='Δ (bps)', color_continuous_scale='RdYlGn_r',
            title=(f"Spread Momentum by Issue Type — Last 24m vs Prior 24m  \n"
                   f"({last_start.date()} → {now_ts.date()} vs {prior_start.date()} → {(last_start - pd.Timedelta(days=1)).date()})"),
            text=mom_issue_sorted['Δ (bps)'].map(lambda v: f"{v:+.1f} bps")
        )
        fig_mom_issue.update_layout(yaxis_title='Δ Spread (bps)', xaxis_title='Issue Type', coloraxis_colorbar=dict(title='Δ bps'))
        fig_mom_issue.update_traces(textposition='outside')
        st.plotly_chart(fig_mom_issue, use_container_width=True)

        with st.expander("Show window stats by Issue Type"):
            st.dataframe(mom_issue_sorted[['Issue Type', 'Last 24m (bps)', 'Count Last 24m',
                                           'Prior 24m (bps)', 'Count Prior 24m', 'Δ (bps)']],
                         use_container_width=True)
    else:
        st.info("Insufficient data (≥2 deals per window) to compute issue-type momentum.")

    # Momentum by Issuer (Top-10 by |Δ|, all issue types)
    mom_issuer, _, _, _ = compute_spread_momentum(base_df, 'Issuer', min_count_per_window=2)
    if not mom_issuer.empty:
        mom_issuer['|Δ|'] = mom_issuer['Δ (bps)'].abs()
        top10_issuer = mom_issuer.sort_values('|Δ|', ascending=False).head(10).sort_values('Δ (bps)')
        fig_mom_issuer = px.bar(
            top10_issuer, x='Issuer', y='Δ (bps)',
            color='Δ (bps)', color_continuous_scale='RdYlGn_r',
            title="Spread Momentum by Issuer — Top 10 by |Δ bps| (Last 24m vs Prior 24m)",
            text=top10_issuer['Δ (bps)'].map(lambda v: f"{v:+.1f} bps")
        )
        fig_mom_issuer.update_layout(yaxis_title='Δ Spread (bps)', xaxis_title='Issuer', coloraxis_colorbar=dict(title='Δ bps'))
        fig_mom_issuer.update_traces(textposition='outside')
        st.plotly_chart(fig_mom_issuer, use_container_width=True)

        with st.expander("Show window stats for selected issuers"):
            st.dataframe(top10_issuer[['Issuer', 'Last 24m (bps)', 'Count Last 24m',
                                       'Prior 24m (bps)', 'Count Prior 24m', 'Δ (bps)']],
                         use_container_width=True)
    else:
        st.info("Insufficient data (≥2 deals per window) to compute issuer momentum.")

    # =============================================================
    # D) Tenor–Spread slope (OLS) — Senior Preferred ONLY, last 24m (min N=3)
    # =============================================================
    st.markdown("### 📐 Tenor–Spread Slope (bps per year) — Senior Preferred (last 24 months)")
    def compute_tenor_spread_slope_sr(df_in: pd.DataFrame, min_count: int = 3):
        d = df_in.copy()
        d['Issue Type Clean'] = d['Issue Type'].astype(str).str.strip()
        d = d[d['Issue Type Clean'] == 'SR Preferred']
        d = d.dropna(subset=['Re-offer Spread', 'Original Tenor', 'Pricing Date'])

        # Last 24 months
        start = pd.Timestamp.today().normalize() - pd.DateOffset(months=24)
        d = d[d['Pricing Date'] >= start]

        results = []
        for issuer, g in d.groupby('Issuer'):
            g = g.dropna(subset=['Original Tenor', 'Re-offer Spread'])
            if len(g) >= min_count and g['Original Tenor'].var() > 0:
                x = g['Original Tenor'].astype(float).values
                y = g['Re-offer Spread'].astype(float).values
                slope, intercept = np.polyfit(x, y, 1)  # slope in bps per tenor-year
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                results.append({'Issuer': issuer, 'N': len(g), 'Slope (bps/yr)': slope, 'R^2': r2})
        out = pd.DataFrame(results).sort_values('Slope (bps/yr)', ascending=False)
        return out, start

    slope_df, slope_start = compute_tenor_spread_slope_sr(base_df, min_count=3)
    if not slope_df.empty:
        # Top 10 by |slope|
        slope_df['|Slope|'] = slope_df['Slope (bps/yr)'].abs()
        top10_slope = slope_df.sort_values('|Slope|', ascending=False).head(10).sort_values('Slope (bps/yr)', ascending=False)

        fig_slope = px.bar(
            top10_slope, x='Issuer', y='Slope (bps/yr)',
            color='Slope (bps/yr)', color_continuous_scale='RdBu_r',
            title=(f"Tenor–Spread Slope by Issuer (SR Preferred) — Top 10 by |slope| "
                   f"({slope_start.date()} → {pd.Timestamp.today().normalize().date()})"),
            text=top10_slope['Slope (bps/yr)'].map(lambda v: f"{v:+.1f} bps/yr")
        )
        fig_slope.update_layout(yaxis_title='Slope (bps per tenor-year)', xaxis_title='Issuer',
                                coloraxis_colorbar=dict(title='bps/yr'))
        fig_slope.update_traces(textposition='outside')
        st.plotly_chart(fig_slope, use_container_width=True)

        with st.expander("Show full slope table (SR Preferred, last 24 months)"):
            st.dataframe(slope_df[['Issuer', 'N', 'Slope (bps/yr)', 'R^2']].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Not enough SR Preferred observations (≥3 per issuer) in the last 24 months to compute tenor–spread slopes.")

    # -------------------------------------------------------------------------
    # Seasonality — Count of deals only (UNFILTERED, no toggles, no normalize)
    # -------------------------------------------------------------------------
    st.markdown("### 🗓️ Issuance Seasonality by Quarter (Count of Deals)")
    season_df = base_df.dropna(subset=['Pricing Date']).copy()
    if season_df.empty:
        st.info("No data available to compute quarter seasonality.")
    else:
        season_df['Pricing Date'] = pd.to_datetime(season_df['Pricing Date'], errors='coerce', dayfirst=True)
        season_df = season_df.dropna(subset=['Pricing Date'])
        season_df['Quarter'] = season_df['Pricing Date'].dt.quarter

        # Aggregate: absolute count per Issuer × Quarter
        agg_df = (season_df.groupby(['Issuer', 'Quarter'])
                            .size()
                            .reset_index(name='Value'))

        # Ensure all quarters 1..4 per issuer exist
        issuers_list = sorted(agg_df['Issuer'].unique().tolist())
        full_index = pd.MultiIndex.from_product([issuers_list, [1, 2, 3, 4]], names=['Issuer', 'Quarter'])
        agg_df = agg_df.set_index(['Issuer', 'Quarter']).reindex(full_index, fill_value=0).reset_index()

        # Clustered bar by Quarter (absolute counts)
        st.markdown("#### Clustered Bars by Quarter (Deal Count)")
        bar_df = agg_df.copy()
        bar_df['Quarter Label'] = bar_df['Quarter'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
        fig_bar = px.bar(
            bar_df,
            x='Quarter Label', y='Value', color='Issuer',
            barmode='group',
            category_orders={'Quarter Label': ['Q1', 'Q2', 'Q3', 'Q4']},
            title="Issuer Seasonality by Quarter — Count of Deals"
        )
        fig_bar.update_layout(yaxis_title='Deals', xaxis_title='Quarter')
        fig_bar.update_traces(marker_line_width=0.3, marker_line_color='rgba(0,0,0,0.4)',
                              hovertemplate="Quarter=%{x}<br>Deals=%{y:.0f}<extra>%{legendgroup}</extra>")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap (Issuer × Quarter) — absolute counts
        st.markdown("#### Heatmap (Issuer × Quarter) — Deal Count")
        heat_pivot = agg_df.pivot(index='Issuer', columns='Quarter', values='Value').reindex(issuers_list)
        heat_pivot = heat_pivot[[1, 2, 3, 4]]
        heat_pivot.columns = ['Q1', 'Q2', 'Q3', 'Q4']
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale='Viridis',
            aspect='auto',
            title="Seasonality Heatmap — Deal Count (Absolute)",
            labels=dict(color="Deals")
        )
        fig_heat.update_layout(xaxis_title='Quarter', yaxis_title='Issuer',
                               coloraxis_colorbar=dict(title='Deals'))
        st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------------------------------------------------------------
# Tables Tab
# -----------------------------------------------------------------------------
with tables_tab:
    if flamboyant:
        st.markdown("### 📋 Tables", unsafe_allow_html=True)

    st.dataframe(filtered_df, use_container_width=True)
    st.write("**Cumulative Issuance by Issuer & Type**")
    st.dataframe(filtered_df.groupby(['Issuer', 'Issue Type'])['Size'].sum().reset_index(), use_container_width=True)
    st.write("**Issuance by Year**")
    st.dataframe(filtered_df.groupby('Year issued')['Size'].sum().reset_index(), use_container_width=True)
    st.write("**Callable Debt by Year**")
    st.dataframe(filtered_df.groupby(filtered_df['FIRST_CALL'].dt.year)['Size'].sum().reset_index(), use_container_width=True)




# -----------------------------------------------------------------------------
# Mount Olympus Tab (podiums & visual effects) — uses UNFILTERED df consistently
# -----------------------------------------------------------------------------
with olympus_tab:
    if flamboyant:
        st.markdown("## 🗻 Mount Olympus: The Pantheon")
        st.markdown("##### ✨ Where Greek bonds ascend to glory! ✨", unsafe_allow_html=True)

    # Fallback if not defined earlier
    UNFILTERED_MARK = globals().get('UNFILTERED_MARK', " †")

    st.info("Note: Mount Olympus podiums use the full dataset (unfiltered) for rankings and highlights.")
    base_df = df  # always full dataset here

    # Ensure normalized issue type exists
    if 'Issue Type Clean' not in base_df.columns:
        base_df['Issue Type Clean'] = base_df['Issue Type'].astype(str).str.strip()

    # --- Colors & podium utilities ------------------------------------------
    MEDAL_EMOJI = {0: "🥇", 1: "🥈", 2: "🥉"}
    MEDAL_COLORS = {'0': '#FFD700', '1': '#C0C0C0', '2': '#CD7F32'}  # gold, silver, bronze

    def podium_df(df_in: pd.DataFrame, value_col: str, top_n: int = 3, ascending: bool = True,
                  extra_cols: list | None = None) -> pd.DataFrame:
        """
        Build a Top-N podium DataFrame from df_in based on value_col.
        - Drops NaNs in value_col
        - Sorts ascending/descending
        - Adds Rank (0..2), Medal, and a two-line Label ('🥇\\nIssuer')
        - Returns only the needed columns (incl. formatted date for preview)
        """
        empty_cols = ['Label', 'RankStr', value_col, 'Issuer', 'Size', 'Coupon', 'Pricing Date (d/m/Y)']
        if df_in is None or df_in.empty or value_col not in df_in.columns:
            return pd.DataFrame(columns=empty_cols)

        # base columns + desired extras (only keep those that exist)
        cols = ['Issuer', value_col]
        if extra_cols:
            cols.extend([c for c in extra_cols if c in df_in.columns])

        d = df_in[cols].copy()
        d = d.dropna(subset=[value_col])
        if d.empty:
            return pd.DataFrame(columns=empty_cols)

        # Friendly date for preview, if Pricing Date present
        if 'Pricing Date' in df_in.columns and 'Pricing Date' in (extra_cols or []):
            # Align by index to pull the right dates
            d['Pricing Date (d/m/Y)'] = pd.to_datetime(
                df_in.loc[d.index, 'Pricing Date'], errors='coerce', dayfirst=True
            ).dt.strftime('%d/%m/%Y')

        # Sort to a Top-N podium
        d = d.sort_values(by=value_col, ascending=ascending).head(top_n).reset_index(drop=True)

        # Add rank/medal + a two‑line label to make the medal stand out
        d['Rank'] = d.index
        d['RankStr'] = d['Rank'].astype(str)
        d['Medal'] = d['Rank'].map(MEDAL_EMOJI)
        d['Label'] = d.apply(lambda r: f"{r['Medal']}\n{r['Issuer']}", axis=1)

        keep = ['Label', 'RankStr', value_col, 'Issuer', 'Size', 'Coupon', 'Pricing Date (d/m/Y)']
        return d[[c for c in keep if c in d.columns]]

    def plot_podium(podium: pd.DataFrame, value_col: str, title: str, yaxis_title: str, value_fmt: str):
        """
        Render a Top-3 podium bar chart with gold/silver/bronze coloring and big medal labels,
        and show the exact rows used as a preview table below.

        IMPORTANT: We pass text as a COLUMN (text='__text') so Plotly maps it row-by-row,
        avoiding the bug where update_traces(text=...) repeats the first value for all traces.
        """
        if podium is None or podium.empty:
            st.info("Not enough data to build a podium.")
            return

        # Preserve podium order on x
        category_order = podium['Label'].tolist()

        # Per-row formatted text bound as a column
        podium = podium.copy()
        podium['__text'] = podium[value_col].apply(lambda v: value_fmt.format(v) if pd.notna(v) else "N/A")

        fig = px.bar(
            podium,
            x='Label',
            y=value_col,
            color='RankStr',
            text='__text',  # <-- per-row text column (CRITICAL FIX)
            color_discrete_map=MEDAL_COLORS,
            title=title
        )
        fig.update_traces(
            textposition='outside',
            marker_line_width=0.6,
            marker_line_color='rgba(0,0,0,0.45)'
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title=yaxis_title,
            xaxis_title='',
            xaxis=dict(categoryorder='array', categoryarray=category_order),
            # Bigger medals (emoji sit on their own line above the issuer)
            xaxis_tickfont=dict(size=24),
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rows preview (helps validate issuer/value alignment)
        with st.expander("Show podium source rows"):
            preview_cols = [c for c in ['Issuer', 'Coupon', 'Size', value_col, 'Pricing Date (d/m/Y)'] if c in podium.columns]
            st.dataframe(podium[preview_cols], use_container_width=True)

    # --- 1) Senior Preferred Champions (Tightest Spreads) --------------------
    st.markdown("### 🏆 Senior Preferred Champions (Tightest Spreads)")
    sr_pref_only = base_df[base_df['Issue Type Clean'] == 'SR Preferred'].copy()
    sr_pref_podium = podium_df(
        sr_pref_only,
        value_col='Re-offer Spread',
        top_n=3, ascending=True,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        sr_pref_podium,
        value_col='Re-offer Spread',
        title='Top 3 Tightest Senior Preferred Spreads' + UNFILTERED_MARK,
        yaxis_title='Spread (bps)',
        value_fmt="{:.1f} bps"
    )

    # --- 2) Tier2 Champions (Tightest Spreads) -------------------------------
    st.markdown("### 🏆 Tier2 Champions (Tightest Spreads)")
    tier2_only = base_df[base_df['Issue Type Clean'] == 'Tier2'].copy()
    tier2_podium = podium_df(
        tier2_only,
        value_col='Re-offer Spread',
        top_n=3, ascending=True,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        tier2_podium,
        value_col='Re-offer Spread',
        title='Top 3 Tightest Tier2 Spreads' + UNFILTERED_MARK,
        yaxis_title='Spread (bps)',
        value_fmt="{:.1f} bps"
    )

    # --- 3) AT1 Champions (Tightest Spreads) ---------------------------------
    st.markdown("### 🏆 AT1 Champions (Tightest Spreads)")
    at1_only = base_df[base_df['Issue Type Clean'] == 'AT1'].copy()
    at1_podium = podium_df(
        at1_only,
        value_col='Re-offer Spread',
        top_n=3, ascending=True,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        at1_podium,
        value_col='Re-offer Spread',
        title='Top 3 Tightest AT1 Spreads' + UNFILTERED_MARK,
        yaxis_title='Spread (bps)',
        value_fmt="{:.1f} bps"
    )

    # --- 4) Highest Spreads — Top 3 (all instruments) ------------------------
    st.markdown("### 🔥 Highest Spreads — Top 3")
    highest_spreads = podium_df(
        base_df,
        value_col='Re-offer Spread',
        top_n=3, ascending=False,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        highest_spreads,
        value_col='Re-offer Spread',
        title='Highest Spreads (Top 3)' + UNFILTERED_MARK,
        yaxis_title='Spread (bps)',
        value_fmt="{:.1f} bps"
    )

    # --- 5) Lowest Spreads — Top 3 (all instruments) -------------------------
    st.markdown("### ❄️ Lowest Spreads — Top 3")
    lowest_spreads = podium_df(
        base_df,
        value_col='Re-offer Spread',
        top_n=3, ascending=True,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        lowest_spreads,
        value_col='Re-offer Spread',
        title='Lowest Spreads (Top 3)' + UNFILTERED_MARK,
        yaxis_title='Spread (bps)',
        value_fmt="{:.1f} bps"
    )

    # --- 6) Shortest Tenors — Top 3 (NC-based tenor, all instruments) --------
    st.markdown("### ⏳ Shortest Tenors — Top 3")
    tenor_nonnull = base_df.dropna(subset=['Original Tenor']).copy()
    shortest_tenors = podium_df(
        tenor_nonnull,
        value_col='Original Tenor',
        top_n=3, ascending=True,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        shortest_tenors,
        value_col='Original Tenor',
        title='Shortest Tenors (Top 3)' + UNFILTERED_MARK,
        yaxis_title='Tenor (yrs)',
        value_fmt="{:.1f} yrs"
    )

    # --- 7) Longest Tenors — Top 3 (NC-based tenor, all instruments) ---------
    st.markdown("### 🧭 Longest Tenors — Top 3")
    longest_tenors = podium_df(
        tenor_nonnull,
        value_col='Original Tenor',
        top_n=3, ascending=False,
        extra_cols=['Pricing Date', 'Size', 'Coupon']
    )
    plot_podium(
        longest_tenors,
        value_col='Original Tenor',
        title='Longest Tenors (Top 3)' + UNFILTERED_MARK,
        yaxis_title='Tenor (yrs)',
        value_fmt="{:.1f} yrs"
    )

    # --- Optional: keep your existing scatter below --------------------------
    # Tightest SR Preferred spread per Original Tenor — unfiltered
    st.markdown("### 🏅 Tightest Senior Preferred by Original Tenor")
    sr_pref_full = base_df[base_df['Issue Type Clean'] == 'SR Preferred'].copy()
    sr_pref_full = sr_pref_full.dropna(subset=['Original Tenor', 'Re-offer Spread'])
    if not sr_pref_full.empty:
        idx_min_per_tenor = sr_pref_full.groupby('Original Tenor')['Re-offer Spread'].idxmin()
        tightest_sr_per_tenor = sr_pref_full.loc[idx_min_per_tenor].sort_values('Original Tenor')
        if 'Pricing Date' in tightest_sr_per_tenor.columns:
            tightest_sr_per_tenor['Pricing Date (d/m/Y)'] = pd.to_datetime(
                tightest_sr_per_tenor['Pricing Date'], errors='coerce', dayfirst=True
            ).dt.strftime('%d/%m/%Y')

        fig_sr_tenor = px.scatter(
            tightest_sr_per_tenor,
            x='Original Tenor',
            y='Re-offer Spread',
            color='Issuer',
            size='Size',
            text='Issuer',
            hover_data={
                'Issuer': True,
                'Original Tenor': ':.2f',
                'Re-offer Spread': ':.1f',
                'Size': ':.0f',
                'Coupon': ':.2f' if 'Coupon' in tightest_sr_per_tenor.columns else False,
                'Pricing Date (d/m/Y)': True if 'Pricing Date (d/m/Y)' in tightest_sr_per_tenor.columns else False,
            },
            title=f'Tightest SR Preferred Spread per Original Tenor{UNFILTERED_MARK}'
        )
        fig_sr_tenor.update_traces(
            textposition='top center',
            marker=dict(opacity=0.9, line=dict(width=0.5, color='rgba(0,0,0,0.45)'))
        )
        fig_sr_tenor.update_layout(
            xaxis_title='Original Tenor (years)',
            yaxis_title='Re-offer Spread (bps)',
            legend_title_text='Issuer'
        )
        st.plotly_chart(fig_sr_tenor, use_container_width=True)
        st.caption("† Based on the tightest (minimum) re-offer spread observed for each distinct Original Tenor across the full dataset.")
    else:
        st.info("No Senior Preferred records with valid Original Tenor and Re-offer Spread found in the full dataset.")


# -----------------------------------------------------------------------------
# Downloads Tab
# -----------------------------------------------------------------------------
with downloads_tab:
    if flamboyant:
        st.markdown("### ⬇️ Downloads", unsafe_allow_html=True)
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Filtered Data (CSV)', data=csv_data, file_name='filtered_data.csv', mime='text/csv')

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
    st.download_button('Download Filtered Data (Excel)', data=output.getvalue(),
                       file_name='filtered_data.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
