import pandas as pd
import re
import time

# Load the Excel file
file_path = 'FCC-Tasks-10222024.xlsx'
df = pd.read_excel(file_path)

start_time = time.time()
print(f"Start time -> \n{start_time}\n")

# Define a regular expression to match only Latin characters and numbers
latin_pattern = r'^[A-Za-z0-9]+$'
def is_latin_only(text):
    # Regular expression for non-Latin characters
    return 'N' if re.search(r'[^\x00-\x7F]', str(text)) else 'Y'

# Apply the function to the 'Short Description' column
df['English'] = df['Summary'].apply(is_latin_only)

# Create a new column to indicate if the "Short Description" contains only Latin characters
#df['English'] = df['Short Description'].apply(lambda x: 'Y' if re.match(latin_pattern, str(x)) else 'N')

# Define a function to check if a string contains only Latin characters and/or numbers

# Save the updated DataFrame back to Excel
df.to_excel('FCC-Tasks-10222024_with_language.xlsx', index=False)

#---------------------------------------- End of the analysis ----------------------------------------#
end_time = time.time()
duration = end_time - start_time
print(f"Duration (in sec) -> \n{duration}")