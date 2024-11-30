import os
import pandas as pd
import re

def format_column_names(df):
    # Use list comprehension to modify column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()
    return df    

# Function to calculate occurrences and percentages
def calculate_filled_fields(dataframe):
    filled_info = []
    total_rows = len(dataframe)
    for col in dataframe.columns:
        filled_count = dataframe[col].notna().sum()  # Count non-NaN values
        filled_percentage = 100-(filled_count / total_rows) * 100
        filled_info.append({
            "Column": col,
            "Filled Count": filled_count,
            "Missing Values %": round(filled_percentage),
        })
    return pd.DataFrame(filled_info)
    
def missing_values_table(df):
    """
    Generate a summary table of missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze for missing values.
    
    Returns:
    pd.DataFrame: A table with columns 'Missing Values' and '% of Total Values', 
                  sorted by '% of Total Values' in descending order. Only columns
                  with missing values are included in the output.
    """
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    
    # Sort the table by percentage of missing descending and round to 1 decimal place
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print("There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def handle_missing_values(df):
    """
    Handles missing values in a DataFrame by filling object-type column NaNs with 'Not Available'
    and dropping remaining columns with missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to process for missing values.

    Returns:
    pd.DataFrame: The cleaned DataFrame with no missing values.
    """

    # Get the initial missing values statistics
    missing_values = missing_values_table(df)
    missing_cols = missing_values.index.tolist()
    print(f"Initial columns with missing values -> \n{missing_cols}\n")

    # Fill missing values for object-type columns
    for col in missing_cols: 
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna('Not Available')

    # Recheck for missing values
    missing_values = missing_values_table(df)
    missing_cols = missing_values.index.tolist()
    print(f"Columns still with missing values after first fill -> \n{missing_cols}\n")

    # Fill remaining missing values for object-type columns
    for col in df.columns:
        if col not in missing_cols and pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna('Not Available')

    # Final check for missing values
    missing_values = missing_values_table(df)
    missing_cols = missing_values.index.tolist()
    print(f"Final columns with missing values -> \n{missing_cols}\n")        

    # Drop columns with remaining missing values
    df = df.drop(missing_cols, axis=1)
    return df

def select_columns_for_analysis(df, available_columns):
    """
    Prompts user to select columns from a list for further analysis.
    
    Parameters:
    df (pd.DataFrame): The DataFrame from which to select columns.
    available_columns (list): List of column names available in the DataFrame.
    
    Returns:
    tuple: A tuple containing:
           - pd.DataFrame: A new DataFrame with only the selected columns.
           - list: List of selected column names.
    """
    print("Available columns for selection:")
    for i, col in enumerate(available_columns, start=1):
        print(f"{i}. {col}")
    
    selected_indices = input("Enter the numbers of the columns you want to select, separated by commas: ")
    try:
        selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
        if all(0 <= idx < len(available_columns) for idx in selected_indices):
            selected_columns = [available_columns[idx] for idx in selected_indices]
            print(f"You have selected columns: {selected_columns}")
            return df[selected_columns], selected_columns
        else:
            print("Invalid selection. Please enter numbers from the list.")
            return None, []
    except ValueError:
        print("Please enter valid numbers separated by commas.")
        return None, []

def get_top_values(df, fields):
    """
    Get the top N most common unique values for specified fields in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    fields (list): A list of column names to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame containing the top N values and their counts for each field.
    """
    top_values_list = []

    # Iterate over each field in the list
    for field in fields:
        # Get the top N most common unique values
        top_15_values = df[field].value_counts().head(15)
        
        # Create a DataFrame from these values
        top_15_df = top_15_values.reset_index()
        top_15_df.columns = [field, 'Count']
        
        # Add a new column to specify which field this data is for
        top_15_df['Field'] = field
        
        # Append this DataFrame to the list
        top_values_list.append(top_15_df)

    # Concatenate all the DataFrames into a single DataFrame
    top_values_df = pd.concat(top_values_list, ignore_index=True)
    return top_values_df

