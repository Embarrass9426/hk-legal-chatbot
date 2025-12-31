from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import json

def extract_keywords(query: str, llm: ChatOpenAI):
    """
    Extracts legal keywords and identifies the target law/case from a user query.
    Returns a dictionary with 'target_law', 'section', and 'keywords'.
    """
    system_prompt = """
    You are a legal research assistant. Analyze the user's query and extract:
    1. 'target_law': The name of the Ordinance or Law (e.g., 'Basic Law', 'Cap 1', 'Employment Ordinance').
    2. 'section': Specific section or article number if mentioned (e.g., 'Section 5', 'Article 23').
    3. 'keywords': A list of 3-5 key terms for searching (e.g., ['right to vote', 'election', 'permanent resident']).
    
    Return ONLY a JSON object.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        response = llm.invoke(messages)
        # Clean the response content in case there's markdown formatting
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        return json.loads(content)
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return {
            "target_law": None,
            "section": None,
            "keywords": [query]
        }
