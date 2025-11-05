import os
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sarvam AI configuration
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
SARVAM_API_URL = os.getenv('SARVAM_API_URL', 'https://api.sarvam.ai/speech-to-text-translate')
SARVAM_MODEL = os.getenv('SARVAM_MODEL', 'saaras:v2.5')

# Hugging Face configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_API_URL = os.getenv('HF_API_URL', 'https://router.huggingface.co/hf-inference/models/d4data/biomedical-ner-all')

app = Flask(__name__)

def transcribe_audio_with_sarvam(audio_file_path: str) -> Dict[str, Any]:
    """Transcribe audio using Sarvam AI"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(audio_file_path), audio_file, 'audio/wav')
            }
            data = {
                'model': SARVAM_MODEL
            }
            headers = {
                'api-subscription-key': SARVAM_API_KEY
            }
            
            response = requests.post(SARVAM_API_URL, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "data": {
                        "text": result.get('transcript', ''),
                        "language_code": result.get('language_code', 'unknown')
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Sarvam API error: {response.status_code}",
                    "details": response.text
                }
    except Exception as e:
        return {
            "success": False,
            "error": "Failed to transcribe audio",
            "details": str(e)
        }

def extract_medical_entities(text: str) -> Dict[str, Any]:
    """Extract medical entities using Hugging Face API"""
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": text
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            raw_entities = response.json()
            
            # Process and categorize entities
            medical_entities = {
                "diseases": [],
                "medications": [],
                "symptoms": [],
                "procedures": [],
                "other": []
            }
            
            # Entity mapping
            entity_category_map = {
                "DISEASE": "diseases",
                "DRUG": "medications",
                "SYMPTOM": "symptoms",
                "PROCEDURE": "procedures",
                "Diagnostic_procedure": "procedures",
                "Medication": "medications",
                "Sign_symptom": "symptoms"
            }
            
            # Process entities
            for entity in raw_entities:
                entity_word = entity.get('word', '')
                entity_group = entity.get('entity', '') or entity.get('entity_group', '')
                
                # Skip if word or group is empty
                if not entity_word or not entity_group:
                    continue
                    
                # Clean up the word
                clean_word = entity_word.strip().rstrip('.,;:!?')
                
                # Skip if word is too short
                if len(clean_word) < 2:
                    continue
                    
                # Map to categories
                category = None
                for key, value in entity_category_map.items():
                    if key in entity_group:
                        category = value
                        break
                
                if category:
                    # Avoid duplicates
                    if clean_word not in medical_entities[category]:
                        medical_entities[category].append(clean_word)
                else:
                    # Add to other
                    medical_entities["other"].append({
                        "word": clean_word,
                        "type": entity_group,
                        "confidence": entity.get('score', 0)
                    })
            
            return {
                "success": True,
                "data": medical_entities
            }
        else:
            return {
                "success": False,
                "error": f"Hugging Face API error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "success": False,
            "error": "Failed to extract medical entities",
            "details": str(e)
        }

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return transcription + medical entities"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No audio file provided"
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No audio file selected"
            }), 400
        
        # Create temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            file.save(tmp_file.name)
            temp_filename = tmp_file.name
        
        try:
            # Step 1: Transcribe audio with Sarvam
            transcription_result = transcribe_audio_with_sarvam(temp_filename)
            
            if not transcription_result["success"]:
                return jsonify({
                    "status": "error",
                    "message": "Failed to transcribe audio",
                    "error_details": transcription_result["error"],
                    "technical_details": transcription_result.get("details", "")
                }), 500
            
            transcription_data = transcription_result["data"]
            
            # Step 2: Extract medical entities
            entities_result = extract_medical_entities(transcription_data["text"])
            
            if not entities_result["success"]:
                return jsonify({
                    "status": "error",
                    "message": "Failed to extract medical entities",
                    "error_details": entities_result["error"],
                    "technical_details": entities_result.get("details", "")
                }), 500
            
            entities_data = entities_result["data"]
            
            # Return successful response
            return jsonify({
                "status": "success",
                "transcription": transcription_data,
                "medical_entities": entities_data,
                "audio_file": os.path.basename(temp_filename)
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "error_details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Medical AI Copilot API is running"
    })

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)