#!/usr/bin/env python3
"""
Script to help rotate the compromised Hugging Face API token
"""

def instructions():
    print("Hugging Face API Token Rotation Instructions")
    print("=" * 50)
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Find the compromised token (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)")
    print("3. Click 'Delete' to remove the compromised token")
    print("4. Click 'New token' to create a new token")
    print("5. Give it a descriptive name like 'medical-ner-app'")
    print("6. Select appropriate permissions (at minimum 'Read')")
    print("7. Copy the new token")
    print("8. Update your .env file with the new token")
    print("9. Set the new token as an environment variable in Render dashboard")
    print("")
    print("For local development:")
    print("  export HF_API_TOKEN=your_new_token_here")
    print("")
    print("For Render deployment:")
    print("  Set HF_API_TOKEN in your Render service environment variables")

if __name__ == "__main__":
    instructions()