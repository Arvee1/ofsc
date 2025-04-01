import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import speech_recognition as sr
import replicate
import pyaudio
import wave
from audiorecorder import audiorecorder

# initialize
r = sr.Recognizer()
# This is in seconds, this will control the end time of the record after the last sound was made
r.pause_threshold = 2

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ofsc_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

collection = client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=embedding_func,
     metadata={"hnsw:space": "cosine"},
 )

# Load VectorDB
# if st.sidebar.button("Load OFSC Facsheets into Vector DB if loading the page for the first time.", type="primary"):
@st.cache_resource
def create_vector():
      with open("ofsc2.txt") as f:
          hansard = f.read()
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=500,
              chunk_overlap=20,
              length_function=len,
              is_separator_regex=False,
          )
           
      texts = text_splitter.create_documents([hansard])
      documents = text_splitter.split_text(hansard)[:len(texts)]
     
      collection.add(
           documents=documents,
           ids=[f"id{i}" for i in range(len(documents))],
      )
      f.close()

create_vector()

# The UI Part
st.title("👨‍💻 Wazzup!!!! What do you want to know about the OFSC?")
# apikey = st.sidebar.text_area("Please enter enter your API Key.")
prompt = st.text_area("Please enter what you want to know about the OFSC and Application process.")

# Load VectorDB
if st.sidebar.button("Load OFSC Facsheets into Vector DB if loading the page for the first time.", type="primary"):
      with open("ofsc2.txt") as f:
          hansard = f.read()
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=500,
              chunk_overlap=20,
              length_function=len,
              is_separator_regex=False,
          )
           
      texts = text_splitter.create_documents([hansard])
      documents = text_splitter.split_text(hansard)[:len(texts)]
     
      collection.add(
          documents=documents,
           ids=[f"id{i}" for i in range(len(documents))],
      )
      f.close()
     
      # number of rows
      # st.write(len(collection.get()['documents']))
      # st.sidebar.write("OFSC Vector DB created. With " + len(collection.get()['documents']) + " rows." )

if st.button("Submit to AI", type="primary"):
     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=20,
     )
     augment_query = str(query_results["documents"])

     result_ai = ""
     # The meta/llama-2-7b-chat model can stream output as it's running.
     # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
     for event in replicate.stream(
         "meta/meta-llama-3-70b-instruct",
         input={
             "top_k": 50,
             "top_p": 0.9,
             "prompt": "Prompt: " + prompt + " " + augment_query,
             "max_tokens": 512,
             "min_tokens": 0,
             "temperature": 0.6,
             "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
             "presence_penalty": 1.15,
             "frequency_penalty": 0.2
         },
     ):
         result_ai = result_ai + (str(event))
     st.write(result_ai)
     

# This is the part where you can verbally ask about stuff
audio = audiorecorder("Click to record", "Click to stop recording")
      
if len(audio) > 0:
     # To play audio in frontend:
     st.audio(audio.export().read())  
     
     # To save audio to a file, use pydub export method:
     audio.export("audio.wav", format="wav")
     print("wav file created")
     # To get audio properties, use pydub AudioSegment properties:
     st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
     
     soundfile = open("audio.wav", "rb")
     text = replicate.run(
          "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
          input={
            "task": "transcribe",
            "audio": soundfile,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
          }
     )
     st.write("what you said: " + text['text'])
     prompt = text['text']

     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=20,
     )
     augment_query = str(query_results["documents"])
     # st.write(augment_query)
     # st.write("###AI Response###")
     # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
     result_ai = ""
     # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
     for event in replicate.stream(
         "meta/meta-llama-3-70b-instruct",
         input={
             "top_k": 50,
             "top_p": 0.9,
             "prompt": "Prompt: " + prompt + " " + augment_query,
             "max_tokens": 512,
             "min_tokens": 0,
             "temperature": 0.6,
             "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
             "presence_penalty": 1.15,
             "frequency_penalty": 0.2
         },
     ):
         result_ai = result_ai + (str(event))
     st.write(result_ai)

     # output = replicate.run(
         # "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71",
         # input={
             # "seed": 0,
             # "text": result_ai,
             # "preset": "fast",
             # "voice_a": "custom_voice",
             # "voice_b": "disabled",
             # "voice_c": "disabled",
             # "cvvp_amount": 0,
             # "custom_voice": "https://replicate.delivery/mgxm/671f3086-382f-4850-be82-db853e5f05a8/nixon.mp3"
         # }
     # )
     # print(output)
