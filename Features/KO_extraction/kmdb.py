import pandas as pd

# Function to extract 'KMDB' solution codes from a column
def extract_kmdb_string(df, column_name):
    """
    Extracts the 'KMDB Solution' string from a specified column and creates a new column with the results.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column to search.

    Returns:
    - pd.DataFrame: The original DataFrame with a new column containing extracted KMDB strings or blank if not found.
    """
    # Define a function to extract the 'KMDB Solution' string
    def extract_kmdb(text):
        if pd.isna(text):
            return None
        match = re.search(r"KMDB Solution\s*:\s*(\w+)(#)?", text)
        if match:
            return match.group(1)
        else:
            return None

    # Apply the extraction function to the entire column and create a new column
    df['KMDB_String'] = df[column_name].apply(extract_kmdb)

    return df

# Function to extract 'KMDB' solution codes from a column
def extract_kmdb_string(df, column_name):
    """
    Extracts the 'KMDB Solution' string from a specified column and creates a new column with the results.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column to search.

    Returns:
    - pd.DataFrame: The original DataFrame with a new column containing extracted KMDB strings or blank if not found.
    """
    # Define a function to extract the 'KMDB Solution' string
    def extract_kmdb(text):
        if pd.isna(text):
            return None
        match = re.search(r"KMDB Solution\s*:\s*(\w+)(#)?", text)
        if match:
            return match.group(1)
        else:
            return None

    # Apply the extraction function to the entire column and create a new column
    df['KMDB_String'] = df[column_name].apply(extract_kmdb)

    return df