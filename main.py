import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

text = "Hey there my name is Vinit Kumar"

encoded = encoder.encode(text=text)

print(f"Encoded: {encoded}")

decoded = encoder.decode(encoded)

print(f"Decoded: {decoded}")
