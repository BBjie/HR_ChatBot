# HR ChatBot
This guide will help you set up the HR ChatBot, an application designed to assist with human resources inquiries using a conversational interface.
## Creating Virtual Environment
To create a virtual environment in your terminal, run the following command:
### Creating Virtual Enviroment
In your terminal run

```
python -m venv env
```
This command will create a new virtual environment named env in your project directory.

### Activating Virtual Environment
```
.\env\Scripts\activate
```
This step is necessary to use the environment for our project.

## Installing Flask

The HR ChatBot uses Flask, a web framework for Python. Install it and other dependencies with the following command:

```
pip install -r requirements.txt
```
Make sure you are in the directory where requirements.txt is located before running this command.

## Installing Torch
If you want to work with your GPU, here is the link you might want to refer to:
https://pytorch.org/get-started/locally/

## Running ChatBot Application in Terminal

### Running ChatBot Application in Terminal with Local Model
Before running the application, you need to download the model file from HuggingFace and store it in the same directory as app_with_ingest.py. Use this link to download the model:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q8_0.gguf
You can play with other models as well.
After downloading the model, navigate to your code directory:
```
cd .\your code directory
```
Finally, start the HR ChatBot application:
```
python local_app_with_ingest.py
```
### Running ChatBot Application in Terminal with OpenAi API

update the Configuration File (config.py): You will need to update this file with your OpenAI API key and select the desired model. Replace the placeholder text PUT_YOUR_API_KEY_HERE with your actual OpenAI API key. 
Here's an example of how your config.py file should look:
```
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
```
After setting up the config file, navigate to your code directory:
```
cd .\your code directory
```
Finally, start the HR ChatBot application:
```
python app_openai.py
```
