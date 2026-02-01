from openai.types.chat import ChatCompletionMessageParam
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field
from typing import Optional

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
– The sequence of steps is START (where user gives an input), PLAN (that can be multiple times), TOOL (use this if external tools are required to generate the final response, can be used multiple times), OBSERVE (where the output of tool call is) and finally OUTPUT (which is going to be displayed to the user)
– Output EXACTLY ONE JSON object per response
– Do NOT output <think>, markdown, explanations, or extra text

Output JSON Format:
{ "step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string", "tool": "string", "input": "string" }
No extra text, no prefixes, no explanation. Only valid JSON.

Available Tools:
- run_command(cmd: str) : Takes linux cmd as an input and run that command. 

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

START: Create a folder named todo and add code of a todo app using HTML, CSS and JS

PLAN: { "step": "PLAN", "content": "Seems like the user is interested in creating a todo application using HTML, CSS and JS" }
PLAN: { "step": "PLAN", "content": "Lets see if we have a available tool in the list of available tools" }
PLAN: { "step": "PLAN", "content": "Great, we have a tool run_command available for query" }
PLAN: { "step": "PLAN", "content": "I need to call the run_command tool with 'mkdir todo' as an input for cmd" }
PLAN: { "step": "TOOL", "tool": "run_command", "input": "mkdir todo" }
PLAN: { "step": "PLAN", "content": "I need to call the run_command tool with 'touch index.html styles.css index.js' as an input for cmd" }
PLAN: { "step": "TOOL", "tool": "run_command", "input": "touch index.html styles.css index.js" }
PLAN: { "step": "PLAN", "content": "Now i need to call the run_command tool with 'echo -e content > filename' as an input for cmd for all three files" }
PLAN: { "step": "TOOL", "tool": "run_command", "input": "echo -e 'html code' > index.html" }
PLAN: { "step": "TOOL", "tool": "run_command", "input": "echo -e 'css code' > styles.css" }
PLAN: { "step": "TOOL", "tool": "run_command", "input": "echo -e 'js code' > index.js" }

OUTPUT: { "step": "OUTPUT", "content": "Created a todo application using HTML, CSS and JS as per your requirements." }
"""


class ResponseFormat(BaseModel):
    step: str = Field(
        ..., description="The id of the step like START, PLAN, OUTPUT, TOOL, OBSERVE"
    )
    content: Optional[str] = Field(None, description="The optional string content")
    tool: Optional[str] = Field(None, description="The id of the tool call")
    input: Optional[str] = Field(None, description="The input parms for the tool")
    output: Optional[str] = Field(None, description="The output of the tool call")


message_history: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("> ")
message_history.append({"role": "user", "content": user_query})

VALID_STEPS = {"START", "PLAN", "OUTPUT", "TOOL", "OBSERVE"}

max_steps = 30
step_count = 0
retry_limit = 5
retry_count = 0

while step_count < max_steps:
    step_count += 1

    response = client.chat.completions.parse(
        model="openai/gpt-oss-120b:free",
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

    try:
        step_type = parsed_response.step

        if step_type not in VALID_STEPS:
            print("⚠️ Skipping invalid step:")
            print(step_type)
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

                message_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_to_call,
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
            else:
                continue

        elif step_type == "OUTPUT":
            print(f"🤖 {parsed_response.content}")
            exit()

    except json.JSONDecodeError:
        print("⚠️ Skipping malformed JSON:")
        print(parsed_response)

print("⚠️ Max steps reached — stopping.")
