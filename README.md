
# Flask App for OpenAI VectorDBQA
This is a Flask app that uses OpenAI's VectorDBQA to answer questions based on a set of documents. The app loads documents from a directory, splits them into chunks, creates OpenAI embeddings, creates a vector store from the documents, and creates a model from the vector store. The app provides two routes: / and /ask. The / route returns an HTML page with a form for submitting questions. The /ask route accepts a POST request with a JSON payload containing a question and returns a JSON response with the answers.



## Installation
1. Clone the repository: git clone https://github.com/username/repo.git
2. Install the dependencies: pip install -r requirements.txt
3. Set the OpenAI API key: export OPENAI_API_KEY=your_api_key


## Usage

1. Start the Flask app: python app.py
2. Open a web browser and go to http://localhost:5000
3. Enter a question in the form and click the "Submit" button
4. The app will return the answers to the question
Configuration

## Configuration
The app can be configured by changing the following variables in app.py:

*  MODEL_NAME: The name of the OpenAI model to use
*  MAX_TOKENS: The maximum number of tokens to use for the model
*  TEMPERATURE: The temperature to use for the model
*  LOADER: The document loader to use
*  CHAR_TEXT_SPLITTER: The character text splitter to use
*  OPENAI_EMBEDDINGS: The OpenAI embeddings to use
*  v_store: The vector store to use
*  model: The VectorDBQA model to use


## License
This code is licensed under the MIT License. See the LICENSE file for details.



## Code
```python
import os
import requests
import nltk
import openai
from flask import Flask, render_template, request, jsonify
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = "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"

# Download NLTK data
nltk.download('averaged_perceptron_tagger')

# Set model parameters
# MODEL_NAME = "text-ada-001"
# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "ada"
MAX_TOKENS = 2048
TEMPERATURE = 0.5

# Load documents from directory
LOADER = DirectoryLoader('docs', glob='**/*.csv')
try:
    docs = LOADER.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    docs = []

if not docs:
    print("No documents found")
    exit()

# Split documents into chunks
CHAR_TEXT_SPLITTER = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
try:
    doc_texts = CHAR_TEXT_SPLITTER.split_documents(docs)
except Exception as e:
    print(f"Error splitting documents: {e}")
    doc_texts = []

if not doc_texts:
    print("No document texts found")
    exit()

# Create OpenAI embeddings
OPENAI_EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create vector store from documents
try:
    v_store = Chroma.from_documents(documents=doc_texts, embedding=OPENAI_EMBEDDINGS)
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit()

# Create model from vector store
v_store = Chroma.from_documents(doc_texts, OPENAI_EMBEDDINGS)
model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=v_store)

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html', form_id='question-form', question_input_id='question', submit_button_value='Submit', answers_div_id='answers')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    try:
        answers = model.run(question)
    except Exception as e:
        print(f"Error running model: {e}")
        answers = []
    return jsonify({'answers': answers})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)