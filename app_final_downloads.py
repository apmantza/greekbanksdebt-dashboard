import streamlit as st
import pandas as pd
from io import BytesIO

# Try importing plotly, handle missing dependency gracefully
try:
    import plotly.express as px
    plotly_available = True
except ModuleNotFoundError:
    st.error("Plotly is not installed. Please add 'plotly' to requirements.txt or install it locally.")
    plotly_available = False

# Load updated data safely
file_path = 'visuals_updated.xlsx'
df = None
try:
    df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')
except FileNotFoundError:
    st.error(f"Data file '{file_path}' not found. Please upload it or check the path.")
except Exception as e:
    st.error(f"Error loading data: {e}")

if df is not None:
    # Convert dates
    for col in ['Pricing Date', 'FIRST_CALL', 'Maturity']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # Sidebar filters
    st.sidebar.header('Filters')
    issuers = st.sidebar.multiselect('Select Issuer(s):', options=df['Issuer'].unique(), default=df['Issuer'].unique())
    issue_types = st.sidebar.multiselect('Select Issue Type(s):', options=df['Issue Type'].unique(), default=df['Issue Type'].unique())
    esg_labels = st.sidebar.multiselect('Select ESG Label(s):', options=df['ESG Label'].unique(), default=df['ESG Label'].unique())

    # Date range filter
    min_date = df['Pricing Date'].min()
    max_date = df['Pricing Date'].max()
    date_range = st.sidebar.slider('Select Pricing Date Range:', min_value=min_date, max_value=max_date, value=(min_date, max_date))

    # Optional filters
    maturity_filter = st.sidebar.multiselect('Select Maturity:', options=df['Maturity'].dropna().unique())

    # Apply filters
    filtered_df = df[
        (df['Issuer'].isin(issuers)) &
        (df['Issue Type'].isin(issue_types)) &
        (df['ESG Label'].isin(esg_labels)) &
        (df['Pricing Date'] >= date_range[0]) &
        (df['Pricing Date'] <= date_range[1])
    ]
    if maturity_filter:
        filtered_df = filtered_df[filtered_df['Maturity'].isin(maturity_filter)]

    # Summary metrics
    st.subheader('Summary Metrics')
    if not filtered_df.empty:
        total_issuances = len(filtered_df)
        avg_spread = round(filtered_df['Re-offer Spread'].mean(), 2)
        min_spread = round(filtered_df['Re-offer Spread'].min(), 2)
        max_spread = round(filtered_df['Re-offer Spread'].max(), 2)
        st.write(f"Total Issuances: {total_issuances}")
        st.write(f"Average Spread: {avg_spread} bps")
        st.write(f"Min Spread: {min_spread} bps")
        st.write(f"Max Spread: {max_spread} bps")
    else:
        st.write("No data available for selected filters.")

    # Charts only if Plotly is available
    if plotly_available and not filtered_df.empty:
        # Trend line grouped by year
        if 'Year issued' in filtered_df.columns:
            trend_df = filtered_df.groupby('Year issued')['Re-offer Spread'].mean().reset_index()
            trend_fig = px.line(trend_df, x='Year issued', y='Re-offer Spread', title='Average Spread per Year')
            st.plotly_chart(trend_fig, use_container_width=True)

        # Scatter plot
        scatter_fig = px.scatter(
            filtered_df,
            x='Pricing Date',
            y='Re-offer Spread',
            color='Issuer',
            hover_data=['Issuer', 'Issue Type', 'ESG Label', 'Maturity', 'FIRST_CALL', 'Pricing Date', 'Re-offer Spread'],
            title="Greek Banks' debt issuances (2019-2025)"
        )
        scatter_fig.update_layout(
            xaxis_title='Pricing Date',
            yaxis_title='Re-offer Spread (bps)',
            yaxis=dict(range=[0, filtered_df['Re-offer Spread'].max() + 50] if not filtered_df.empty else [0, 1000]),
            legend_title_text='Issuer'
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # Download filtered data
    st.subheader('Download Filtered Data')
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(label='Download CSV', data=csv, file_name='filtered_data.csv', mime='text/csv')

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
        st.download_button(label='Download Excel', data=output.getvalue(), file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Display filtered data table
    st.write('Filtered Data:', filtered_df)
