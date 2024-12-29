import requests
import json
from typing import Dict, Any
import os

def get_financial_statement(tickers: list[str], api_key: str = "6c72ec47e6f87dfba40040b696df1e1b") -> Dict[str, Any]:
    if not tickers:
        return {}

    # Load existing data if file exists
    existing_data = {}
    if os.path.exists("financial_statements.json"):
        with open("financial_statements.json", "r") as f:
            existing_data = json.load(f)

    base_url = "https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported"
    params = {
        "period": "annual",
        "limit": 50,
        "apikey": api_key
    }
    
    results = existing_data.copy()  # Start with existing data
    
    for ticker in tickers:
        # Skip if we already have this ticker's data
        if ticker in results:
            continue
            
        response = requests.get(f"{base_url}/{ticker}", params=params)
        
        if not response.ok:
            continue
            
        results[ticker] = response.json()
            
    with open("financial_statements.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results


# Load tickers
with open('tickers.json', 'r') as file:
    tickers = json.load(file)

# Process the specified range
get_financial_statement(tickers[200:450])