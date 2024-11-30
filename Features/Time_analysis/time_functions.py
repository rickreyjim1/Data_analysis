#Import Required Libraries
import pandas as pd

def extract_date_time_features(df, fields):
    """
    Extracts year, month, hour, day of week, day name, day, and week number 
    from specified date columns and returns a subset with only the new columns.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        fields (list): A list of column names containing the date and time values.

    Returns:
        pandas.DataFrame: A new DataFrame containing only the extracted features.
    """
    new_columns = {}

    for field in fields:
        if field not in df.columns:
            print(f"Warning: Column '{field}' not found in the DataFrame. Skipping.")
            continue
        
        # Ensure the field is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[field]):
            try:
                df[field] = pd.to_datetime(df[field])
            except Exception as e:
                print(f"Error converting column '{field}' to datetime: {e}")
                continue

        # Extract and save new columns to the dictionary
        new_columns[f'{field}_year'] = df[field].dt.year
        new_columns[f'{field}_month'] = df[field].dt.month_name()
        new_columns[f'{field}_hour'] = df[field].dt.hour
        new_columns[f'{field}_day_of_week'] = df[field].dt.dayofweek
        new_columns[f'{field}_day_name'] = df[field].dt.day_name()
        new_columns[f'{field}_day'] = df[field].dt.day
        new_columns[f'{field}_week_number'] = df[field].dt.isocalendar().week

    # Convert the dictionary to a DataFrame
    new_columns_df = pd.DataFrame(new_columns)
    return new_columns_df