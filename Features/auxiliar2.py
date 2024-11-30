def plot_word_count_distribution(df, field, output_dir, use_log=False):
    """
    Plots an enhanced histogram of word count distribution.

    Parameters:
    - df: pandas DataFrame containing the data.
    - field: str, the name of the column to analyze.
    - use_log: bool, whether to use a logarithmic scale on the y-axis.
    """

    # Calculate max, min, and average
    max_value = df[field].max()
    min_value = df[field].min()
    average_value = df[field].mean()
    median_value = df[field].median()

    print(f"Max: {max_value}, Min: {min_value}, Average: {average_value}, Median: {median_value}")
    num_bins=10
    # Create dynamic bins
    bins = np.linspace(min_value, max_value, num_bins + 1)

    # Set plot style
    sns.set(style="whitegrid")

    # Create the histogram plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df[field], bins=bins, kde=True, color='skyblue', edgecolor='black')

    # Add vertical lines for mean and median
    plt.axvline(average_value, color='red', linestyle='--', label=f'Mean: {average_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')

    # Logarithmic scale option
    if use_log:
        plt.yscale('log')

    # Add titles and labels
    plt.title(f'Enhanced Distribution of {field} Lengths')
    plt.xlabel(f'{field} Length Bins')
    plt.ylabel('Occurrences')
    plt.legend()

    # Show gridlines
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent clipping

    # Save the figure with a unique filename
    plt.savefig(f'{output_dir}/{field}_distribution.png')

    # Clear the current figure to avoid overlap with the next plot
    plt.clf()


