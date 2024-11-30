import os
import pandas as pd

def create_folder(folder_path):
    """
    Create a new folder at the specified path.
    
    Args:
        folder_path (str): The full path of the folder to create.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating folder: {e}")

def select_file(folder_path):
    """
    Lists all Excel and CSV files in the specified folder and prompts the user to select one.

    Parameters:
    folder_path (str): The path to the folder containing the files.

    Returns:
    str: The name of the selected file or None if an invalid selection was made.
    """
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter only Excel and CSV files
    excel_files = [f for f in files if f.endswith('.xlsx')]
    csv_files = [f for f in files if f.endswith('.csv')]

    # Combine the lists
    all_files = excel_files + csv_files

    # Display the files with a numbered list
    print("The next are the file(s) in the root folder:")
    for idx, file in enumerate(all_files, start=1):
        print(f"{idx}. {file}")

    # Prompt the user for input
    try:
        selection = int(input("Enter the number of the file you want to select: ")) - 1
        if 0 <= selection < len(all_files):
            selected_file = all_files[selection]
            return selected_file
        else:
            print("Invalid selection. Please enter a number from the list.")
            return None
    except ValueError:
        print("Please enter a valid number.")
        return None

        