import requests  # For making HTTP requests to the local LLM API

def call_llm_diagnostic(score, result, location="ATM-123", atm_type="NCR-Model-X"):
    """
    Sends ATM image comparison results to a locally running LLM (e.g., via LM Studio)
    to get a diagnostic explanation of the tampering likelihood, reasoning, and recommended actions.

    Args:
        score (float): Similarity score from the Siamese model (0 to 1)
        result (str): Classification result ("Tampered" or "Clean")
        location (str): Identifier or name of the ATM location
        atm_type (str): Model or type of ATM hardware (e.g., NCR, Diebold)

    Returns:
        str: LLM-generated diagnostic explanation and recommendation
    """

    # Prompt formatted with input score, result, ATM details
    prompt = f"""
You are an ATM fraud expert. An ATM image was compared to a clean reference.

Similarity score: {score:.4f}
Status: {result}

ATM Type: {atm_type}
Location: {location}

Please explain if this is likely tampering, describe the cause, and recommend an action.
"""

    # Construct the payload for the LM Studio HTTP API
    payload = {
        "model": "mythomax-l2-13b",  # Your selected local model
        "messages": [
            {"role": "system", "content": "You are an ATM fraud analyst assistant."},  # Role setting
            {"role": "user", "content": prompt}  # Actual diagnostic prompt
        ],
        "temperature": 0.7,         # Adds randomness to the response
        "max_tokens": -1,           # Use full context limit
        "stream": False             # Return full response at once (non-streaming)
    }

    # Send POST request to the LM Studio API endpoint
    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload)

    # Extract and return the LLM's textual response
    return response.json()["choices"][0]["message"]["content"]