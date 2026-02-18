from openai.types.chat import ChatCompletionMessageParam
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
from typing import Optional
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

eleven_labs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not eleven_labs_api_key:
    raise RuntimeError("ELEVENLABS_API_KEY not set")

eleven_client = ElevenLabs(api_key=eleven_labs_api_key)


def speak(text: str):
    audio = eleven_client.text_to_speech.convert(
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_multilingual_v2",
        text=text,
    )
    play(audio)


def run_command(cmd: str):
    exit_code = os.system(cmd)
    print(f"Ran command: {cmd} with exit code {exit_code}")
    return f"Ran command with exit code {exit_code}"


available_tools = {"run_command": run_command}


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
– The sequence of steps is START, PLAN, TOOL, OBSERVE, OUTPUT
– Output EXACTLY ONE JSON object per response
– Do NOT output anything except valid JSON

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string", "tool": "string", "input": "string" }

Available Tools:
- run_command(cmd: str)
"""


class ResponseFormat(BaseModel):
    step: str
    content: Optional[str] = None
    tool: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None


message_history: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

r = sr.Recognizer()

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_api_key,
)

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    r.pause_threshold = 2

    print("Speak something...")
    audio = r.listen(source)

    print("Processing audio...")
    stt = r.recognize_google(audio)

    print("You said:", stt)

user_query = stt
message_history.append({"role": "user", "content": user_query})

VALID_STEPS = {"START", "PLAN", "OUTPUT", "TOOL", "OBSERVE"}

max_steps = 30
step_count = 0
retry_limit = 5
retry_count = 0

while step_count < max_steps:
    step_count += 1

    response = client.chat.completions.parse(
        model="gemini-2.5-flash",
        messages=message_history,
        response_format=ResponseFormat,
    )

    parsed_response = response.choices[0].message.parsed

    if not parsed_response:
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
    step_type = parsed_response.step

    if step_type not in VALID_STEPS:
        print("⚠️ Skipping invalid step:", step_type)
        continue

    message_history.append(
        {"role": "assistant", "content": json.dumps(parsed_response.model_dump())}
    )

    if step_type == "START":
        print(f"🔥 {parsed_response.content}")

    elif step_type == "PLAN":
        print(f"🧠 {parsed_response.content}")

    elif step_type == "TOOL":
        tool_to_call = parsed_response.tool
        tool_input = parsed_response.input

        if isinstance(tool_to_call, str) and isinstance(tool_input, str):
            print(f"🛠️: {tool_to_call} {tool_input}")

            tool_response = available_tools[tool_to_call](tool_input)

            # FIX: keep inside normal assistant message flow
            message_history.append(
                {
                    "role": "assistant",
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
        print(f"🤖 {parsed_response.content}")

        if parsed_response.content:
            speak(parsed_response.content)

        break  # FIX: instead of exit()

print("⚠️ Max steps reached — stopping.")

