import base64
from openai import OpenAI, AsyncOpenAI

from PIL import Image
import io
import os
import json_repair
import re
from dotenv import load_dotenv

load_dotenv()

async def continue_openai(messages, new_message, model_name="gpt-4o-mini"):
    messages.append(
        {
            "role" : "assistant",
            "content" : new_message
        }
    )
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model=model_name,
        messages= messages,
        max_tokens=500
    )

    llm_response = response.choices[0].message.content

    return llm_response


async def generate_openai(image_path, prompt, model_name="gpt-4o-mini"):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}",
                        },
                    },
                ],
            }
        ]

    response = await client.chat.completions.create(
        model=model_name,
        messages= messages,
        max_tokens=300
    )

    llm_response = response.choices[0].message.content

    messages.append(
        {
            "role": "assistant",
            "content": llm_response
        }
    )

    return llm_response, messages


def extract_json_string(text: str) -> str:
    # Try to find JSON enclosed in triple backticks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)

    if json_match:
        json_str = json_match.group(1)
    else:
        # If not found, try to find JSON without backticks
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            json_str = json_match.group(0)
        else:
            return ""

    return json_str