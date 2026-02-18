from openai import OpenAI
import speech_recognition as sr
import os
from dotenv import load_dotenv
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


def main():
    r = sr.Recognizer()

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_api_key,
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

    reply = response.choices[0].message.content

    if reply is None:
        print("Failed to get response.")
        return

    print("Reply:", reply)

    speak(reply)


main()
