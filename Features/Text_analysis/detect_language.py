import pandas as pd
import requests
import uuid
import time
# Keys
trans_key = "21f3ef5da3c940ac9d19cba322321e07"
trans_location = "eastus"

trans_headers = {
    'Ocp-Apim-Subscription-Key': trans_key,
    'Ocp-Apim-Subscription-Region': trans_location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}
language_endpoint = "https://api.cognitive.microsofttranslator.com/"
detection_url = language_endpoint + "detect?api-version=3.0"

start_time = time.time()
print(f"Start time -> \n{start_time}\n")

def translate_input(userquestion):
    body = [{"text": userquestion}]

    # Send the POST request for detection
    #response = requests.post(detection_url, params=None, headers=trans_headers, json=body, verify=False)
    request = requests.post(detection_url, params=None, headers=trans_headers, json=body)
    request.raise_for_status()  # Check if the request was successful
    
    # Parse the response JSON
    response = request.json()
    
    # Extracting translation, language and score
    language = response[0]['language']
    score = response[0]['score']
    return {'language':language,'score':score}


# Read the Excel file
input_file_path = "New_Dump_with_language.xlsx"
df = pd.read_excel(input_file_path)
#df = df.head(2000)  # Limit to 2000 rows for testing
#df = df.sample(2000)  # Limit to 3000 rows for testing

# Initialize new columns for the translation results
df['Language'] = ""
df['Score'] = ""

# Iterate over the rows and apply the translation function to the 'Short Description' column
short_description = df['short_description']

# Translate each description one by one
for index, row in df.iterrows():
  text_to_translate = row["short_description"]
  translation_result = translate_input(text_to_translate)
  df.at[index, 'Language'] = translation_result["language"]
  df.at[index, 'Score'] = translation_result["score"]
  print(f"Index = {index}")
  
# Save the updated DataFrame to a new Excel file
output_file_path = "New_data_Not_english_with_language.xlsx"
df.to_excel(output_file_path, index=False)

print("Translation processing completed and saved to:", output_file_path)

#---------------------------------------- End of the analysis ----------------------------------------#
end_time = time.time()
duration = end_time - start_time
print(f"Duration (in sec) -> \n{duration}")