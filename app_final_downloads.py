import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

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

    # Date range filter
    valid_dates = df['Pricing Date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().to_pydatetime()
        max_date = valid_dates.max().to_pydatetime()
        date_range = st.sidebar.slider('Select Pricing Date Range:', min_value=min_date, max_value=max_date, value=(min_date, max_date))
    else:
        st.warning("No valid Pricing Date values found.")
        date_range = (None, None)

    maturity_filter = st.sidebar.multiselect('Select Maturity:', options=df['Maturity'].dropna().unique())

    # Apply filters
    filtered_df = df[(df['Issuer'].isin(issuers)) & (df['Issue Type'].isin(issue_types)) & (df['ESG Label'].isin(esg_labels))]
    if date_range[0] and date_range[1]:
        filtered_df = filtered_df[(filtered_df['Pricing Date'] >= date_range[0]) & (filtered_df['Pricing Date'] <= date_range[1])]
    if maturity_filter:
        filtered_df = filtered_df[filtered_df['Maturity'].isin(maturity_filter)]

    # Summary metrics
    st.subheader('Summary Metrics')
    if not filtered_df.empty:
        st.write(f"Total Issuances: {len(filtered_df)}")
        st.write(f"Average Spread: {round(filtered_df['Re-offer Spread'].mean(), 2)} bps")
    else:
        st.write("No data available for selected filters.")

    # Trend toggle
    trend_option = st.sidebar.radio("Select Trend View:", ["Overall Trend", "Per Bank Trend"])
    if not filtered_df.empty:
        if trend_option == "Overall Trend":
            trend_df = filtered_df.groupby('Year issued')['Re-offer Spread'].mean().reset_index()
            trend_fig = px.line(trend_df, x='Year issued', y='Re-offer Spread', title='Average Spread per Year')
        else:
            trend_df = filtered_df.groupby(['Year issued', 'Issuer'])['Re-offer Spread'].mean().reset_index()
            trend_fig = px.line(trend_df, x='Year issued', y='Re-offer Spread', color='Issuer', markers=True,
                                title='Average Spread per Year by Bank')
        st.plotly_chart(trend_fig, use_container_width=True)

    # Scatter chart type toggle
    chart_type = st.sidebar.radio("Select Scatter Chart Type:", ["Scatter", "Bar"])
    if chart_type == "Scatter":
        scatter_fig = px.scatter(filtered_df, x='Pricing Date', y='Re-offer Spread', color='Issuer',
                                 hover_data={'Issuer': True, 'Issue Type': True, 'ESG Label': True, 'Maturity': True, 'Maturity.1': True, 'FIRST_CALL': True},
                                 title="Greek Banks' Debt Issuances")
    else:
        scatter_fig = px.bar(filtered_df, x='Issuer', y='Re-offer Spread', color='Issuer',
                             title="Average Spread by Issuer")
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Liability profiles
    st.subheader('Liability Profile by Bank (Issuance Size per Year)')
    if 'FIRST_CALL' in filtered_df.columns and 'Size' in filtered_df.columns:
        call_df = filtered_df.dropna(subset=['FIRST_CALL'])
        if not call_df.empty:
            call_df['Call Year'] = call_df['FIRST_CALL'].dt.year
            liability_df_bank = call_df.groupby(['Call Year', 'Issuer'])['Size'].sum().reset_index()
            liability_fig_bank = px.bar(liability_df_bank, x='Call Year', y='Size', color='Issuer', barmode='group',
                                        title='Liability Profile: Issuance Size per Year by Bank')
            st.plotly_chart(liability_fig_bank, use_container_width=True)

    st.subheader('Liability Profile by Issue Type (Issuance Size per Year)')
    if 'FIRST_CALL' in filtered_df.columns and 'Size' in filtered_df.columns:
        call_df_type = filtered_df.dropna(subset=['FIRST_CALL'])
        if not call_df_type.empty:
            call_df_type['Call Year'] = call_df_type['FIRST_CALL'].dt.year
            liability_df_type = call_df_type.groupby(['Call Year', 'Issue Type'])['Size'].sum().reset_index()
            liability_fig_type = px.bar(liability_df_type, x='Call Year', y='Size', color='Issue Type', barmode='group',
                                        title='Liability Profile: Issuance Size per Year by Issue Type')
            st.plotly_chart(liability_fig_type, use_container_width=True)

    # Download buttons for charts and data
    st.subheader('Download Charts and Data')
    if not filtered_df.empty:
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(label='Download Filtered Data (CSV)', data=csv, file_name='filtered_data.csv', mime='text/csv')

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
        st.download_button(label='Download Filtered Data (Excel)', data=output.getvalue(), file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Chart downloads
        for fig, name in [(scatter_fig, 'scatter_chart.png'), (trend_fig, 'trend_chart.png'), (liability_fig_bank, 'liability_bank.png'), (liability_fig_type, 'liability_type.png')]:
            if fig:
                img_bytes = fig.to_image(format="png")
                st.download_button(label=f"Download {name}", data=img_bytes, file_name=name, mime="image/png")

except FileNotFoundError:
    st.error(f"Data file '{file_path}' not found. Please ensure it is in the GitHub repo.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
