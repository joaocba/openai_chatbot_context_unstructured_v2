##
# Uses LangChain, LlamaIndex and UnstructuredReader loader to prepare the dataset for the vector
# Load an unstructured local file and vector it
# Save the vectorized file on disk
# Has a logger

## Dependecies:
# pip install flask
# pip install llama_index
# pip install langchain


from flask import Flask, render_template, request
from pathlib import Path
import os
import logging
import datetime

from llama_index import GPTSimpleVectorIndex, download_loader, LLMPredictor, SimpleDirectoryReader, PromptHelper
from langchain import OpenAI



# create logs folder if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# get current date and time for log filename
log_filename = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# configure logger
logging.basicConfig(filename=log_filename, level=logging.INFO)


# set OpenAI API
# os.environ["OPENAI_API_KEY"] = ''
# in case it is already defined on windows path variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


## context data config ##
def load_data():
    UnstructuredReader = download_loader("UnstructuredReader")
    loader = UnstructuredReader()
    documents = loader.load_data(file=Path('./content/empresas.txt'))
    return documents


## AI config ##
def get_llm_predictor():
    # define AI model
    model = "text-davinci-003"
    # define creativity in the response
    creativity = 0
    # number of completions to generate
    completions = 1
    # set number of output tokens
    num_outputs = 250
    # params for LLM (Large Language Model)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature = creativity,
            model_name = model,
            max_tokens = num_outputs,
            #top_p=0,
            #frequency_penalty=0,
            #presence_penalty=0.7,
            n = completions
        )
    )
    return llm_predictor


def get_prompt_helper():
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 250
    # set maximum chunk overlap
    max_chunk_overlap = 40
    # set chunk size limit
    chunk_size_limit = 1000

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit = chunk_size_limit
    )
    return prompt_helper


# method to generate response querying the indexed data
def generate_response(prompt, index):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(prompt, response_mode="compact")
    return response.response


# array to store conversations
conversation = ["You are a virtual assistant and you speak portuguese."]    # define initial role

# load indexed data
documents = load_data()
index = GPTSimpleVectorIndex(documents, llm_predictor=get_llm_predictor())

# verify if file already exists
if not os.path.exists('index.json') or not index.save_to_disk:
    index.save_to_disk('index.json') # save indexed data on local file


app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    # get user input
    user_input = request.args.get("msg") + '\n'
    response = ''
    if user_input:
        conversation.append(f"{user_input}")

        # get conversation history
        prompt = "\n".join(conversation[-3:])

        # generate AI response based on indexed data
        response = generate_response(prompt, index) + '\n'

        # add AI response to conversation
        conversation.append(f"{response}")

        # log conversation
        with open(log_filename, "a") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"AI: {response}\n\n")

        # log conversation using logger
        logging.info(f"User: {user_input}")
        logging.info(f"AI: {response}")

    return response if response else "Sorry, I didn't understand that."



if __name__ == "__main__":
    app.run()
