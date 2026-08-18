import pandas as pd
from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

def simple_analytics(df: pd.DataFrame):

    '''
    A Simple function to show null values in dataset and dtypes of columns
    '''

    null_values = df.isna().sum()
    dtypes = df.dtypes

    print(null_values)
    print('\n')
    print(dtypes)


def load_raw_datset(filepath: str) -> pd.DataFrame:

    '''
    Lodaing raw csv with correct dtypes in pandas. Parsing "InvoiceDate" Column to datetime column
    '''

    df = pd.read_csv(filepath_or_buffer=filepath, parse_dates=["InvoiceDate"])
    return df


def clean_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Cleans the following transactions:
    1) Removing all rows where customer id is NaN or null value.
    2) Removes Cancellations (starting with 'C' in Invoice string) and negative quantities.
    3) Removes zero/negative price items.
    4) Calculates TotalSpend = Quantity * Price
    '''

    df = df.dropna(subset=["Customer ID"]).copy()

    # Converting CustomerID column to int dtype
    df["Customer ID"] = df["Customer ID"].astype(int) 

    # Filtering Cancelltions and Negative quantitites
    df = df[~df["Invoice"].astype(str).str.startswith('C')]
    df = df[df["Quantity"]>0]

    # Remove zero/negative price 
    df = df[df["Price"]>0]

    # Making TotalSpend column
    df["TotalSpend"] = df["Quantity"]*df["Price"]

    return df


def create_customer_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Splits the dataset into Year 1 and Year 2. Year 1 data will be used to gropuby customerID and then using aggregates to define new columns/features.

    Year 2 dataset will be used to define target Matrix.
    '''

    split_date = pd.Timestamp('2010-12-01') # It will take Year 1 from Dec-1-2009 to Nov-30-2010. So year 2 will be from Dec-1-2010 to Dec-9-2011. 

    year1_df = df[df["InvoiceDate"]<split_date].copy()
    year2_df = df[df["InvoiceDate"]>=split_date].copy()


    # Using year1_df to import make Feature Matrix X

    max_year1_date = year1_df["InvoiceDate"].max()

    basket_size = year1_df.groupby(by=["Customer ID", "Invoice"])["Quantity"].sum().reset_index()
    avg_basket_size = basket_size.groupby(by="Customer ID")["Quantity"].mean().rename("AvgBasketSize")

    features = year1_df.groupby("Customer ID").agg(
        Recency=('InvoiceDate', lambda x: (max_year1_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalSpend', 'sum'),
        Tenure=('InvoiceDate', lambda x: (max_year1_date - x.min()).days)
    ).reset_index()

    features = features.merge(avg_basket_size, on="Customer ID", how="left")


    # Using year2_df to make Target Matrix

    targets = year2_df.groupby("Customer ID").agg(
        Target_LTV=('TotalSpend','sum')
    ).reset_index()


    dataset = pd.merge(features, targets, on="Customer ID", how="left") # This is like SQL JOin of 2 tables, the join used is left join because this is left join there might be case that right table has null value targets in this case, so we will have to fill them

    dataset["Target_LTV"] = dataset["Target_LTV"].fillna(0.0)

    # Creating a new column 'churn' if customer spends in year2, which means Targer_LTV=somevalue then it will be 0 if cutomer does not spend anything in year 2 then churn=1

    dataset["Target_churn"] = (dataset["Target_LTV"] == 0).astype(int)

    return dataset


if __name__ == "__main__":

    raw_df_path = RAW_DATA_DIR / "online_retail_raw.csv"

    raw_df = load_raw_datset(raw_df_path)
    cleaned_df = clean_raw_dataset(raw_df)

    ml_df = create_customer_features_and_targets(cleaned_df)

    output_path = PROCESSED_DATA_DIR / "customer_features_targets.csv"

    ml_df.to_csv(output_path, index=False)

    print("Dataset Processing is done, check data/processed Folder\n")
    print(ml_df.head())