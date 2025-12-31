from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import json

async def extract_keywords(query: str, llm: ChatOpenAI):
    """
    Extracts legal keywords and identifies the target law/case from a user query.
    Returns a dictionary with 'target_law', 'section', and 'keywords'.
    """
    system_prompt = """
    You are a legal research assistant specializing in Hong Kong Employees' Compensation.
    Analyze the user's query and extract:
    1. 'target_law': The name of the Ordinance (usually 'Employees' Compensation Ordinance' or 'Cap 282').
    2. 'section': Specific section number if mentioned.
    3. 'keywords': A list of 3-5 key terms for searching (e.g., ['workplace injury', 'insurance', 'medical expenses', 'permanent disability']).
    
    Return ONLY a JSON object.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    try:
        response = await llm.ainvoke(messages)
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
