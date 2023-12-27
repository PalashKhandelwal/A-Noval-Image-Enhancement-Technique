import pandas as pd
import plotly.express as px
import re

# Read the Excel file
excel_file_path = 'gantt.xlsx'
df = pd.read_excel(excel_file_path)

# Define a function to extract date information
def extract_dates(date_string):
    matches = re.findall(r'\b\d{1,2}\w{0,2}\s\w+\b', date_string)
    if len(matches) == 2:
        return matches
    else:
        return [date_string, date_string]  # return original string if not matched

# Apply the function to extract start and end dates
df[['Start Date', 'End Date']] = df['Date'].apply(extract_dates).apply(pd.Series)

# Create Gantt chart using the string dates
fig = px.timeline(df, x_start='Start Date', x_end='End Date', y='Task', title='Gantt Chart', category_orders={'Task': df['Task'].unique()})

# Customize the appearance of the Gantt chart if needed
fig.update_yaxes(categoryorder='total ascending')
fig.update_layout(xaxis_title='Timeline')

# Show the Gantt chart
fig.show()
