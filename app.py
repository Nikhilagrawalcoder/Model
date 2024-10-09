from flask import Flask, request, jsonify
import os
import re
import base64
import json
import pandas as pd
import requests
import time
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from collections import OrderedDict
from difflib import get_close_matches

app = Flask(__name__)

# Configuration
JSON_FILE_PATH = r"C:\Users\nikhi\Downloads\output.json"
EXCEL_FILE_PATH = r"C:\Users\nikhi\Downloads\output.xlsx"
BUFFER_SIZE = 10
MAX_SUGGESTIONS = 3
API_KEY = "AIzaSyCQrYGVRTNivr4Dh_xhJLkVovy6kDEFhKY"

# Helper function to load acronyms from a JSON file
def load_acronyms(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}.")
        return {}

# Acronym Lookup Class
class AcronymLookup:
    def __init__(self, json_file_path):
        self.acronyms_dict = load_acronyms(json_file_path)
        self.acronym_buffer = OrderedDict()

    def get_suggestions(self, acronym):
        return get_close_matches(acronym, self.acronyms_dict.keys(), n=MAX_SUGGESTIONS, cutoff=0.6)

    def lookup_acronym(self, acronym):
        if not acronym:
            return {"error": "No acronym provided"}

        acronym = acronym.upper()
        
        if acronym in self.acronym_buffer:
            meaning = self.acronym_buffer[acronym]
            self.acronym_buffer.move_to_end(acronym)
            return {"acronym": acronym, "meaning": meaning, "source": "buffer"}
        
        meaning = self.acronyms_dict.get(acronym)
        
        if meaning:
            if len(self.acronym_buffer) >= BUFFER_SIZE:
                self.acronym_buffer.popitem(last=False)
            self.acronym_buffer[acronym] = meaning
            return {"acronym": acronym, "meaning": meaning, "source": "json"}
        
        suggestions = self.get_suggestions(acronym)
        if suggestions:
            return {
                "message": "Acronym not found",
                "suggestions": [{"acronym": s, "meaning": self.acronyms_dict[s]} for s in suggestions]
            }
        return {"message": "Acronym not found and no similar acronyms available"}

# Helper function to download and process image
def download_and_process_image(image_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_url, timeout=10, headers=headers, stream=True)
        response.raise_for_status()
        
        image_content = response.content
        
        if not response.headers.get('content-type', '').startswith('image/'):
            raise ValueError("URL does not point to a valid image")
        
        image = Image.open(BytesIO(image_content))
        
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95, optimize=True)
        image_bytes = buffered.getvalue()
        
        if len(image_bytes) == 0:
            raise ValueError("Processed image is empty")
            
        return image_bytes
        
    except requests.RequestException as e:
        raise Exception(f"Error downloading image: {str(e)}")
    except (IOError, ValueError) as e:
        raise Exception(f"Error processing image: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

# Function to analyze document from a URL
def analyze_document_from_url(image_url):
    try:
        genai.configure(api_key=API_KEY)

        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        image_data = download_and_process_image(image_url)
        image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode('utf-8')}

        prompt_template = """Analyze this image and extract all visible text and information. Return it in the following JSON format:
        {
            "Name_of_Treating_Doctor": "",
            "Contact_Number": "",
            "Nature_of_Illness": "",
            "Relevant_Critical_Findings": "",
            "Duration_of_Ailment": {"Days": ""},
            "Date_of_First_Consultation": "",
            "Past_History": "",
            "Provisional_Diagnosis": "",
            "ICD_10_Code": "",
            "Proposed_Treatment": {"Medical_Management": false, "Surgical_Management": false, "Intensive_Care": false, "Investigation": false, "Non_Allopathic_Treatment": false}
        }"""

        prompt_parts = [prompt_template, image_part]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt_parts)
                if not response or not response.text:
                    raise ValueError("Empty response from model")

                response_text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
                response_text = re.sub(r'^```\s*|\s*```$', '', response_text).replace('\n', ' ').replace('\r', '').strip()

                parsed_json = json.loads(response_text)
                return json.dumps(parsed_json, indent=2)

            except json.JSONDecodeError as je:
                if attempt == max_retries - 1:
                    raise

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)
                continue

    except Exception as e:
        raise Exception(f"Error in analyzing document: {str(e)}")

# Endpoint to process image from URL
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()

        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image URL provided'}), 400

        image_url = data['image_url']
        if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid image URL format'}), 400

        extracted_text = analyze_document_from_url(image_url)
        extracted_data = json.loads(extracted_text)

        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_data, json_file, indent=4)

        excel_update_success = update_excel_with_new_data(extracted_data)

        return jsonify({
            'status': 'success',
            'data': extracted_data,
            'excel_updated': excel_update_success
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to update Excel with new data
def update_excel_with_new_data(new_data):
    try:
        if os.path.exists(EXCEL_FILE_PATH):
            existing_df = pd.read_excel(EXCEL_FILE_PATH)
        else:
            existing_df = pd.DataFrame()

        new_df = pd.json_normalize(new_data)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_excel(EXCEL_FILE_PATH, index=False)
        return True
    except Exception as e:
        return False

