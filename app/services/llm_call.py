from openai import AsyncOpenAI
from app.core.config import setting
from app.services.clean_llm_response import extract_json_from_llm

async def call_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"):
    """
    Universal async function to call OpenAI LLM.
    
    Args:
        system_prompt (str): The system role instruction.
        user_prompt (str): The user's input or task.
        model (str): The LLM model to use. Default: gpt-4o-mini
    
    Returns:
        str: The model's text response.
    """
    try:
        async with AsyncOpenAI(api_key=setting.openai_api_key) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1000
            )

        response = response.choices[0].message.content.strip()
        json_response = extract_json_from_llm(response)
        return json_response

    except Exception as e:
        print(f"LLM call failed: {e}")
        raise e

    
