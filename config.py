configs={
    "max_new_tokens": 350,             # Maximum number of new tokens to generate
    "top_p": 0.8,                      # sampling probability threshold
    "temperature": 0.5,                # Controls randomness in generation
    # "presence_penalty": 1.0,           # Penalty for repeating content that is already present
    # "frequency_penalty": 1.0,          # Penalty for repeating the same line of thought
    # "logprobs": 20,                    # Returns the top 20 token probabilities
    # "echo": False                       # Includes the input prompt in the output
}



# config.py
API_KEY = "PUT_YOUR_API_KEY_HERE"
openai_configs = {
    "temperature": 0.8,
    "max_tokens": 350,
    "top_p": 0.5,
     # frequency_penalty=0.5,  
    # presence_penalty=0.5,   
    "model": "gpt-4",
    "openai_api_key": API_KEY
}
