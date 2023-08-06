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
import openai
from flask_caching import Cache

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
OPENAI_API_KEY = ""

# Download NLTK data
nltk.download('averaged_perceptron_tagger')

# Set model parameters
# MODEL_NAME = "text-ada-001"
# MODEL_NAME = "gpt-3.5-turbo"
# MODEL_NAME = "ada"
# MODEL_NAME = "text-davinci-003"

MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Load documents from directory
LOADER = DirectoryLoader('docs',glob='**/*.txt')
try:
    docs = LOADER.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    docs = []

if not docs:
    print("No documents found")
    exit()

# Split documents into chunks
CHAR_TEXT_SPLITTER = CharacterTextSplitter(chunk_size=12000, chunk_overlap=0)
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
model = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens=200), chain_type="stuff", vectorstore=v_store)

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
