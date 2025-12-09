import requests

def chatgpt_response(text, model='gpt-4o-2024-08-06'):
    ###
    api_key = ""
    api_url = ""
    headers = {
        """
        """
    }
    data = {
        'model': model,
        'messages': [{
            'role': 'system',
            'content': ''},
            {
                'role': 'user',
                'content': text
            }]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data).json()
        return response['choices'][0]['message']['content']
    except Exception as e:
        # Catch any errors in the network request
        print(f"Request failed: {e}")
        return "Error: Request failed due to a network issue."

if __name__ == "__main__":
    print(chatgpt_response("hello"))
