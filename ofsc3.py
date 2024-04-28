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

# The UI Part
st.title("👨‍💻 Wazzup!!!! What do you want to know about the OFSC?")
# apikey = st.sidebar.text_area("Please enter enter your API Key.")
prompt = st.text_area("Please enter what you want to know about the OFSC Accreditation process.")

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
          n_results=100,
     )
     augment_query = str(query_results["documents"])
    
     # client_AI = OpenAI(api_key=apikey)
     client_AI = OpenAI(api_key=st.secrets["api_key"])
     response = client_AI.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[
         {
           "role": "system",
           "content": "You are a friendly assistant."
         },
         {
           "role": "user",
           "content": augment_query + " Prompt: " + prompt
         }
       ],
       temperature=0.1,
       max_tokens=1000,
       top_p=1
     )
     
     st.write(response.choices[0].message.content)

if st.button("Say something", type="primary"):
     chunk = 1024  # Record in chunks of 1024 samples
     sample_format = pyaudio.paInt16  # 16 bits per sample
     channels = 2
     fs = 44100  # Record at 44100 samples per second
     seconds = 3
     filename = "output.wav"
     
     p = pyaudio.PyAudio()  # Create an interface to PortAudio
     
     print('Recording')
     
     stream = p.open(format=sample_format,
                     channels=channels,
                     rate=fs,
                     frames_per_buffer=chunk,
                     input=True)
     
     frames = []  # Initialize array to store frames
     
     # Store data in chunks for 3 seconds
     for i in range(0, int(fs / chunk * seconds)):
         data = stream.read(chunk)
         frames.append(data)
     
     # Stop and close the stream
     stream.stop_stream()
     stream.close()
     # Terminate the PortAudio interface
     p.terminate()
     
     print('Finished recording')
     
     # Save the recorded data as a WAV file
     wf = wave.open(filename, 'wb')
     wf.setnchannels(channels)
     wf.setsampwidth(p.get_sample_size(sample_format))
     wf.setframerate(fs)
     wf.writeframes(b''.join(frames))
     wf.close()

     soundfile = open("output.wav.wav", "rb")
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

     # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
     result_ai = ""
     # The meta/llama-2-7b-chat model can stream output as it's running.
     for event in replicate.stream(
            "meta/llama-2-7b-chat",
            input={
                "top_k": 0,
                "top_p": 1,
                "prompt": prompt,
                "temperature": 0.75,
                "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                "length_penalty": 1,
                "max_new_tokens": 800,
                "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                "presence_penalty": 0
            },
     ):
        result_ai = result_ai + (str(event))
     
     st.write(result_ai)
