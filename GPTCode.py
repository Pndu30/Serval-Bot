from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def get_message(message):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "assistant", "content": str(message)},
    ],
  )
  return response.choices[0].message.content