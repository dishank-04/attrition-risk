import pandas as pd
from src.paths import RAW_DATA_DIR

print("Starting the conversion...")

input_file = RAW_DATA_DIR / "online_retail_II.xlsx"
output_file = RAW_DATA_DIR / "online_retail_raw.csv"

df1 = pd.read_excel(input_file, sheet_name="Year 2009-2010")
df2 = pd.read_excel(input_file, sheet_name='Year 2010-2011')

# Converting excel files to .csv to save time while loading 

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv(output_file, index=False)

print("Conversion Ended...")
