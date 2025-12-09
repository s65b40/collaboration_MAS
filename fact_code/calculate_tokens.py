import tiktoken

def calculate_tokens(text, model="gpt-4o"):

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model(model)
    
    tokens = enc.encode(text)
    token_count = len(tokens)

    return token_count

if __name__ == "__main__":
    input_text = "Hello, how are you?"
    result = calculate_tokens(input_text)

    print(f"count of input tokens: {result}")

