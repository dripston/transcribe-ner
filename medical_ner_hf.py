import requests
import json
import os
from typing import Dict, List, Optional, Any

class MedicalNERHF:
    def __init__(self, hf_api_token: str):
        """
        Initialize Medical NER with Hugging Face API
        """
        self.hf_api_token = hf_api_token
        # Updated endpoint as per the error message
        self.hf_api_url = "https://router.huggingface.co/hf-inference/models/d4data/biomedical-ner-all"
        self.headers = {"Authorization": f"Bearer {hf_api_token}"}
    
    def extract_entities(self, text: str) -> Optional[List[Dict]]:
        """
        Extract medical entities from text using Hugging Face API
        """
        try:
            # For NER models, we don't need aggregation_strategy parameter
            payload = {
                "inputs": text
            }
            
            print("Sending text to Hugging Face Medical NER API...")
            response = requests.post(
                self.hf_api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                entities = response.json()
                print("Medical entities extracted successfully!")
                return entities
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return None
    
    def _reconstruct_entities(self, entities: List[Dict], original_text: str) -> List[Dict]:
        """
        Reconstruct split entities by looking at their positions in the original text
        """
        if not entities:
            return entities
            
        reconstructed = []
        i = 0
        
        while i < len(entities):
            current = entities[i].copy()
            
            # Get the start and end positions
            start_pos = current.get('start', 0)
            end_pos = current.get('end', start_pos)
            
            # Look for consecutive entities that might be part of the same word
            j = i + 1
            while j < len(entities):
                next_entity = entities[j]
                next_start = next_entity.get('start', 0)
                
                # If the next entity starts right after the current one ends, 
                # and they're likely part of the same word
                if next_start <= end_pos + 2:  # Allow for space or punctuation
                    # Extend the end position
                    end_pos = max(end_pos, next_entity.get('end', next_start))
                    j += 1
                else:
                    break
            
            # Extract the actual text from the original text
            if start_pos < len(original_text) and end_pos <= len(original_text):
                actual_text = original_text[start_pos:end_pos].strip()
                # Remove any trailing punctuation that might have been included
                actual_text = actual_text.rstrip('.,;:!?')
                current['word'] = actual_text
            
            reconstructed.append(current)
            i = j  # Skip the entities we've processed
            
        return reconstructed
    
    def process_medical_text(self, transcription: str) -> Dict[str, Any]:
        """
        Process medical transcription and categorize entities
        """
        # Extract entities using Hugging Face API
        raw_entities = self.extract_entities(transcription)
        
        # Handle error case
        if raw_entities is None:
            return {"error": "Failed to extract entities - API error"}
        
        # Handle case where response is an error message (not a list)
        if isinstance(raw_entities, dict) and "error" in raw_entities:
            return {"error": f"API Error: {raw_entities.get('error', 'Unknown error')}"}
        
        # If we got an empty response or unexpected format
        if not isinstance(raw_entities, list):
            return {"error": "Unexpected API response format"}
        
        # Reconstruct split entities
        processed_entities = self._reconstruct_entities(raw_entities, transcription)
        
        # Categorize entities
        medical_entities: Dict[str, Any] = {
            "diseases": [],
            "medications": [],
            "symptoms": [],
            "procedures": [],
            "other": []
        }
        
        # Common medical entity mappings
        entity_category_map = {
            "DISEASE": "diseases",
            "DRUG": "medications",
            "SYMPTOM": "symptoms",
            "PROCEDURE": "procedures",
            "Diagnostic_procedure": "procedures",
            "Medication": "medications",
            "Sign_symptom": "symptoms"
        }
        
        for entity in processed_entities:
            entity_word = entity.get('word', '')
            entity_group = entity.get('entity', '') or entity.get('entity_group', '')
            
            # Skip if word or group is empty
            if not entity_word or not entity_group:
                continue
                
            # Clean up the word
            clean_word = entity_word.strip().rstrip('.,;:!?')
            
            # Skip if word is too short or just punctuation
            if len(clean_word) < 2:
                continue
                
            # Map to our categories
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
                # Add to other with entity type info
                medical_entities["other"].append({
                    "word": clean_word,
                    "type": entity_group,
                    "confidence": entity.get('score', 0)
                })
        
        return medical_entities

def main():
    # Get Hugging Face API token from environment variables
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    
    if not HF_API_TOKEN:
        print("Error: HF_API_TOKEN environment variable not set")
        print("Please set your Hugging Face API token in the environment variables")
        print("Example: export HF_API_TOKEN=your_token_here")
        return
    
    # Initialize the Medical NER
    print("Initializing Medical NER System with Hugging Face API...")
    ner_model = MedicalNERHF(HF_API_TOKEN)
    
    # Example medical text (this would come from your Sarvam transcription)
    sample_text = "Patient presents with hypertension and diabetes. Prescribed metformin and lisinopril. Recommended echocardiogram and blood tests."
    
    print(f"\nProcessing sample medical text:\n{sample_text}")
    
    # Extract medical entities
    entities = ner_model.process_medical_text(sample_text)
    
    # Display results
    print("\nMedical Entities Extracted:")
    print("=" * 30)
    
    # Check if there was an error
    if "error" in entities:
        print(f"Error: {entities['error']}")
        return
    
    # Display categorized entities
    for category, items in entities.items():
        if items:
            print(f"\n{category.capitalize()}:")
            if category == "other":
                for item in items:
                    print(f"  - {item['word']} ({item['type']}) - Confidence: {item['confidence']:.2f}")
            else:
                for item in items:
                    print(f"  - {item}")

if __name__ == "__main__":
    main()