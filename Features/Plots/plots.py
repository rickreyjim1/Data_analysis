#Import Required Libraries
 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Function to plot top values for a list of fields

def plot_top_values(top_values_df, output_dir):
    """
    Generate and save bar plots for the top values of each field.

    Parameters:
    top_values_df (pd.DataFrame): DataFrame containing the top values for each field.
    output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each field
    for field in top_values_df['Field'].unique():
        # Filter the DataFrame for the current field
        field_df = top_values_df[top_values_df['Field'] == field]

        # Plot the bar chart
        plt.box(on=True)
        plt.figure(figsize=(10, 6))
        plt.bar(field_df[field], field_df['Count'], color='skyblue')
        plt.title(f'Top Most Common Values in {field}')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')

        # Add values on top of the bars
        for i, value in enumerate(field_df['Count']):
            plt.text(i, value, str(value), ha='center', va='bottom')

        plt.tight_layout()

        # Save the plot with a unique filename
        output_path = os.path.join(output_dir, f"{field}_top_values.png")
        plt.savefig(output_path)
        
        # Clear the figure to avoid overlap
        plt.clf()
        
def plot_word_count_distribution(df, field, output_dir):
    """
    Plots a histogram of word count distribution.

    Parameters:
    - df: pandas DataFrame containing the data.
    - field: str, the name of the column to analyze.
    -
    """

    # Calculate max, min, and average
    max_value = df[field].max()
    min_value = df[field].min()
    average_value = df[field].mean()
    median_value = df[field].median()

    print(f"Max: {max_value}, Min: {min_value}, Average: {average_value}, Median: {median_value}")
    num_bins=10
    # Calculate the bin width based on the number of bins and data range
    bin_width = (max_value - min_value)/num_bins

    # Create bin edges using numpy.linspace
    bins = np.linspace(min_value, max_value + bin_width, num_bins + 1)

    plt.hist(df[field], bins=bins, edgecolor='skyblue')

    # Add vertical lines for mean and median
    plt.axvline(average_value, color='red', linestyle='--', label=f'Mean: {average_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')

    # Add titles and labels
    plt.title(f'Distribution of {field} Lengths')
    plt.xlabel(f'{field} Length Bins')
    plt.ylabel('Occurrences')
    plt.legend()

    # Show gridlines
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent clipping

    # Save the figure with a unique filename
    plt.savefig(f'{output_dir}/{field}_distribution.png')

    # Close the plot to avoid display overload when running in loops
    plt.close()  
    # Clear the current figure to avoid overlap with the next plot
    plt.clf()

def plot_and_save_time_related_data(time_columns, output_dir):
    """
    Plots and saves time-related data for the given columns.

    Parameters:
    - time_columns: list, a list of column names to plot and analyze (e.g., ['created_day_name', 'created_month']).
    - output_dir: str, directory where the plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define custom orderings for days and months
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']

    # Loop through each selected column
    for column in time_columns:
        if column.endswith("_day_of_week"):
            continue  # Skip columns ending with "_day_of_week"

        if column not in time_columns.columns:
            print(f"Warning: Column '{column}' not found in the DataFrame. Skipping.")
            continue

        # Handle categorical ordering for specific time-related columns
        if 'day_name' in column:
            time_columns[column] = pd.Categorical(time_columns[column], categories=days_order, ordered=True)
        elif 'month' in column:
            time_columns[column] = pd.Categorical(time_columns[column], categories=months_order, ordered=True)

        # Special handling for week numbers
        if 'week_number' in column:
            # Ensure all weeks (1 to 52/53) are included, even if no data for some weeks
            full_week_range = pd.Series(range(1, 54), name=column)  # Covers all possible week numbers
            grouped_data = time_columns[column].value_counts(sort=False).reindex(full_week_range, fill_value=0)
        else:
            # Group by the selected column and count occurrences
            grouped_data = time_columns[column].value_counts(sort=False).sort_index()

        # Create the plot
        plt.figure(figsize=(10, 6))
        grouped_data.plot(kind='line', marker='o', color='skyblue')

        # Add values on top of the points
        for i, value in enumerate(grouped_data):
            plt.text(i, value, str(value), ha='center', va='bottom', fontsize=9)

        # Add titles and labels
        plt.title(f'Occurrences by {column}', fontsize=16)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Adjust layout and save the figure
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'by_{column}.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to avoid display overload when running in loops
        # Clear the current figure to avoid overlap with the next plot
        plt.clf()