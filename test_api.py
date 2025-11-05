import requests

def test_api():
    """Test the Medical AI Copilot API"""
    # API endpoint
    url = "http://localhost:5000/process_audio"
    
    # Path to your test audio file
    audio_file_path = "recording.wav"
    
    try:
        # Open and send the audio file
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(url, files=files)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print("=" * 50)
            print(f"Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"\nTranscription: {result['transcription']['text']}")
                print(f"Language: {result['transcription']['language_code']}")
                
                print("\nMedical Entities:")
                entities = result['medical_entities']
                for category, items in entities.items():
                    if items:
                        print(f"  {category.capitalize()}: {items}")
            else:
                print(f"Error: {result['message']}")
                if 'error_details' in result:
                    print(f"Details: {result['error_details']}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_api()