from openai import OpenAI
from app.core.config import setting

client = OpenAI(api_key=setting.openai_api_key)

def generate_thread_title(first_user_message: str) -> str:
    prompt = f"""
    You are a chat assistant. A new conversation has started.
    The first user message is: "{first_user_message}"
    Generate a short, descriptive title for this conversation in 3-5 words.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    title = response.choices[0].message.content.strip().strip('"')
    return title

if __name__=="__main__":
    title = generate_thread_title("Hi")
    print("thread title --------------", title)