import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime

st.title("Greek Banks Debt Dashboard")

file_path = 'visuals_updated.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')

    # Convert date columns safely
    for col in ['Pricing Date', 'FIRST_CALL', 'Maturity.1']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # Sidebar filters
    st.sidebar.header('Filters')
    issuers = st.sidebar.multiselect('Select Issuer(s):', options=df['Issuer'].dropna().unique(), default=df['Issuer'].dropna().unique())
    issue_types = st.sidebar.multiselect('Select Issue Type(s):', options=df['Issue Type'].dropna().unique(), default=df['Issue Type'].dropna().unique())
    esg_labels = st.sidebar.multiselect('Select ESG Label(s):', options=df['ESG Label'].dropna().unique(), default=df['ESG Label'].dropna().unique())

    # Date range filter for pricing
    valid_dates = df['Pricing Date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().to_pydatetime()
        max_date = valid_dates.max().to_pydatetime()
        date_range = st.sidebar.slider('Select Pricing Date Range:', min_value=min_date, max_value=max_date, value=(min_date, max_date))
    else:
        st.warning("No valid Pricing Date values found.")
        date_range = (None, None)

    # Toggle for call date range
    call_valid_dates = df['FIRST_CALL'].dropna()
    if not call_valid_dates.empty:
        call_min_date = call_valid_dates.min().to_pydatetime()
        call_max_date = call_valid_dates.max().to_pydatetime()
        call_date_range = st.sidebar.slider('Select Call Date Range:', min_value=call_min_date, max_value=call_max_date, value=(call_min_date, call_max_date))
    else:
        call_date_range = (None, None)

    maturity_filter = st.sidebar.multiselect('Select Maturity:', options=df['Maturity'].dropna().unique())

    # Apply filters
    filtered_df = df[(df['Issuer'].isin(issuers)) & (df['Issue Type'].isin(issue_types)) & (df['ESG Label'].isin(esg_labels))]
    if date_range[0] and date_range[1]:
        filtered_df = filtered_df[(filtered_df['Pricing Date'] >= date_range[0]) & (filtered_df['Pricing Date'] <= date_range[1])]
    if call_date_range[0] and call_date_range[1]:
        filtered_df = filtered_df[(filtered_df['FIRST_CALL'] >= call_date_range[0]) & (filtered_df['FIRST_CALL'] <= call_date_range[1])]
    if maturity_filter:
        filtered_df = filtered_df[filtered_df['Maturity'].isin(maturity_filter)]

    # Summary metrics
    st.subheader('Summary Metrics')
    if not filtered_df.empty:
        total_issuances = len(filtered_df)
        cumulative_issuance = filtered_df['Size'].sum() if 'Size' in filtered_df.columns else None
        cumulative_by_issuer_type = filtered_df.groupby(['Issuer','Issue Type'])['Size'].sum() if 'Size' in filtered_df.columns else None
        avg_issuance_per_year = cumulative_issuance / len(filtered_df['Year issued'].unique()) if cumulative_issuance else None

        next_year = datetime.now().year + 1
        maturing_next_year = filtered_df[filtered_df['Maturity.1'].dt.year == next_year] if 'Maturity.1' in filtered_df.columns else pd.DataFrame()
        debt_next_year_size = maturing_next_year['Size'].sum() if 'Size' in maturing_next_year.columns else None
        avg_spread_next_year = maturing_next_year['Re-offer Spread'].mean() if not maturing_next_year.empty else None

        st.write(f"Total Issuances: {total_issuances}")
        if cumulative_issuance:
            st.write(f"Cumulative Issuance Size: {cumulative_issuance:,.0f}")
        if cumulative_by_issuer_type is not None:
            st.write("Cumulative Issuance by Issuer and Issue Type:")
            st.write(cumulative_by_issuer_type)
        if avg_issuance_per_year:
            st.write(f"Average Issuance per Year: {avg_issuance_per_year:,.0f}")
        if debt_next_year_size:
            st.write(f"Debt Maturing Next Year: {debt_next_year_size:,.0f}")
        if avg_spread_next_year:
            st.write(f"Average Spread of Debt Maturing Next Year: {avg_spread_next_year:.2f} bps")

        # Extra visual: Debt maturing next year by issuer (move before liability profiles)
        if not maturing_next_year.empty:
            next_year_fig = px.bar(maturing_next_year.groupby('Issuer')['Size'].sum().reset_index(),
                                   x='Issuer', y='Size', color='Issuer',
                                   title=f'Debt Maturing in {next_year} by Issuer',
                                   labels={'Size': 'Issuance Size'})
            st.plotly_chart(next_year_fig, use_container_width=True)

    else:
        st.write("No data available for selected filters.")

    # First visual: Per Bank Trend
    if not filtered_df.empty:
        trend_df = filtered_df.groupby(['Year issued', 'Issuer'])['Re-offer Spread'].mean().reset_index()
        trend_fig = px.line(trend_df, x='Year issued', y='Re-offer Spread', color='Issuer', markers=True,
                            title='Average Spread per Year by Bank')
        st.plotly_chart(trend_fig, use_container_width=True)

    # Second visual: Scatter only
    scatter_fig = px.scatter(filtered_df, x='Pricing Date', y='Re-offer Spread', color='Issuer',
                             hover_data={'Issuer': True, 'Issue Type': True, 'ESG Label': True, 'Maturity': True, 'Maturity.1': True, 'FIRST_CALL': True},
                             title="Greek Banks' Debt Issuances")
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Liability profiles (start from current year)
    st.subheader('Liability Profiles')
    current_year = datetime.now().year
    if 'FIRST_CALL' in filtered_df.columns and 'Size' in filtered_df.columns:
        call_df = filtered_df.dropna(subset=['FIRST_CALL'])
        call_df['Call Year'] = call_df['FIRST_CALL'].dt.year
        call_df = call_df[call_df['Call Year'] >= current_year]
        if not call_df.empty:
            liability_df_bank = call_df.groupby(['Call Year', 'Issuer'])['Size'].sum().reset_index()
            liability_fig_bank = px.bar(liability_df_bank, x='Call Year', y='Size', color='Issuer', barmode='group',
                                        title='Liability Profile: Issuance Size per Year by Bank')
            st.plotly_chart(liability_fig_bank, use_container_width=True)

            liability_df_type = call_df.groupby(['Call Year', 'Issue Type'])['Size'].sum().reset_index()
            liability_fig_type = px.bar(liability_df_type, x='Call Year', y='Size', color='Issue Type', barmode='group',
                                        title='Liability Profile: Issuance Size per Year by Issue Type')
            st.plotly_chart(liability_fig_type, use_container_width=True)

    # Download buttons for charts and data (HTML format)
    st.subheader('Download Charts and Data')
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(label='Download Filtered Data (CSV)', data=csv, file_name='filtered_data.csv', mime='text/csv')

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
        st.download_button(label='Download Filtered Data (Excel)', data=output.getvalue(), file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        for fig, name in [(scatter_fig, 'scatter_chart.html'), (trend_fig, 'trend_chart.html'), (liability_fig_bank, 'liability_bank.html'), (liability_fig_type, 'liability_type.html'), (next_year_fig, 'next_year_chart.html')]:
            if fig:
                html_bytes = fig.to_html().encode('utf-8')
                st.download_button(label=f"Download {name}", data=html_bytes, file_name=name, mime="text/html")

except FileNotFoundError:
    st.error(f"Data file '{file_path}' not found. Please ensure it is in the GitHub repo.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
