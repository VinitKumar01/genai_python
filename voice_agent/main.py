from openai import OpenAI
import speech_recognition as sr
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")


def main():
    r = sr.Recognizer()

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # noise cancellation
        r.pause_threshold = 2  # start if user pauses for 2 sec

        print("Speak something...")
        audio = r.listen(source)

        print("Processing audio...")
        # speech to text
        stt = r.recognize_google(audio)

        print("You said:", stt)

    SYSTEM_PROMPT = """
    You are an expert voice agent. You are given the transcript of what user have said using voice.
    
    You need to output as if you are an  voice agent and whatever you speak will be converted back to audio using AI and played back to user.
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stt},
        ],
    )

    print(response.choices[0].message.content)


main()
