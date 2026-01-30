import requests
import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://127.0.0.1",
        "X-Title": "hello-world-test",
    },
)


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"


available_tools = {"get_weather": get_weather}


# Chain of thought prompts are the type of prompts which uses multiple steps to respond to a problem query.
SYSTEM_PROMPT = """
You're an expert AI Assistant in resolving user queries using chain of thought.
You work on START, PLAN and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be multiple steps.
Once you think enough PLAN has been done, finally you can give an OUTPUT.
You can also call tools if required from the list of available tools.
For every tool call wait for the OBSERVE step which is the output from the called tool.

Rules:
– Strictly follow the given JSON output format
– Only run one step at a time
– The sequence of steps is START (where user gives an input), PLAN (that can be multiple times), TOOL (use this if external tools are required to generate the final response, can be used multiple times), OBSERVE (where the output of tool call is) and finally OUTPUT (which is going to be displayed to the user)
– Output EXACTLY ONE JSON object per response
– Do NOT output <think>, markdown, explanations, or extra text

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string", "tool": "string", "input": "string" }
No extra text, no prefixes, no explanation. Only valid JSON.

Available Tools:
- get_weather(city: str) : Takes city name as an input and returns the weather info about the city. 

Example 1:

START: Hey, Can you solve 2 + 3 * 5 / 10

PLAN: { "step": "PLAN", "content": "Seems like user is interested in math problem" }
PLAN: { "step": "PLAN", "content": "looking at the problem, we should solve this using BODMAS method" }
PLAN: { "step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here" }
PLAN: { "step": "PLAN", "content": "first we must multiply 3 * 5 which is 15" }
PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 15 / 10" }
PLAN: { "step": "PLAN", "content": "We must perform divide that is 15 / 10 = 1.5" }
PLAN: { "step": "PLAN", "content": "Now the new equation is 2 + 1.5" }
PLAN: { "step": "PLAN", "content": "Now finally lets perform the add 3.5" }
PLAN: { "step": "PLAN", "content": "Great, we have solved and finally left with 3.5 as ans" }

OUTPUT: { "step": "OUTPUT", "content": "3.5" }

Example 2:

START: What is the weather of Delhi?

PLAN: { "step": "PLAN", "content": "Seems like the user is interested in getting the weather of Delhi in India" }
PLAN: { "step": "PLAN", "content": "Lets see if we have a available tool in the list of available tools" }
PLAN: { "step": "PLAN", "content": "Great, we have a tool get_weather available for query" }
PLAN: { "step": "PLAN", "content": "I need to call the get_weather tool with delhi as an input for city" }
PLAN: { "step": "TOOL", "tool": "get_weather", "input": "delhi" }
PLAN: { "step": "OBSERVE", "tool": "get_weather", "input": "delhi", "output": "The temperature of delhi is cloudy with 20 C" }
PLAN: { "step": "PLAN", "content": "Great i got the weather info about delhi" }

OUTPUT: { "step": "OUTPUT", "content": "The current weather for delhi is 20 C with cloudy sky." }
"""

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("> ")
message_history.append({"role": "user", "content": user_query})


def extract_json_objects(text):
    objects = []
    stack = []
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    objects.append(text[start : i + 1])
                    start = None
    return objects


VALID_STEPS = {"START", "PLAN", "OUTPUT", "TOOL", "OBSERVE"}

max_steps = 30
step_count = 0
retry_limit = 5
retry_count = 0

while step_count < max_steps:
    step_count += 1

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=message_history,
    )

    raw_response = response.choices[0].message.content.strip()

    raw_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.S).strip()

    json_objects = extract_json_objects(raw_response)

    if not json_objects:
        retry_count += 1
        print("⚠️ Model returned no JSON — retrying...")

        if retry_count >= retry_limit:
            print("❌ Too many failures. Stopping.")
            break

        message_history.append(
            {
                "role": "system",
                "content": "You FAILED. Output ONE valid JSON step only.",
            }
        )
        continue

    retry_count = 0

    for obj in json_objects:
        try:
            parsed = json.loads(obj)
            step_type = parsed.get("step")
            content = parsed.get("content", "")

            if step_type not in VALID_STEPS:
                print("⚠️ Skipping invalid step:")
                print(obj)
                continue

            message_history.append({"role": "assistant", "content": obj})

            if step_type == "START":
                print(f"🔥 {content}")

            elif step_type == "PLAN":
                print(f"🧠 {content}")

            elif step_type == "TOOL":
                tool_to_call = parsed.get("tool")
                tool_input = parsed.get("input")
                print(f"🛠️: {tool_to_call} {tool_input}")

                tool_response = available_tools[tool_to_call](tool_input)

                message_history.append(
                    {
                        "role": "developer",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_to_call,
                                "input": tool_input,
                                "output": tool_response,
                            }
                        ),
                    }
                )

            elif step_type == "OUTPUT":
                print(f"🤖 {content}")
                exit()

        except json.JSONDecodeError:
            print("⚠️ Skipping malformed JSON:")
            print(obj)

print("⚠️ Max steps reached — stopping.")
