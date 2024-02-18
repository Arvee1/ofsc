__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

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
apikey = st.sidebar.text_area("Please enter enter your API Key.")
prompt = st.text_area("Please enter what you want to know about the OFSC Accreditation process.")


# Load VectorDB
if st.sidebar.button("Load OFSC Facsheets into Vector DB if loading the page for the first time.", type="primary"):
      with open("5. Fact Sheet - How to appeal a decision of the Federal Safety Commissioner.txt") as f:
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
     
      with open("Applying the Scheme to Directly funded building work accessible version_0.txt") as f:
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

      with open("Applying the Scheme to Indirectly funded building work accessible version.txt") as f:
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

      with open("Fact Sheet - Advice for Funding Recipients_0.txt") as f:
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
     
      with open("Fact Sheet - Advice to Agencies.txt") as f:
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

      with open("Fact Sheet - Complaints Process_0.txt") as f:
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

      with open("Fact Sheet - Corrective Action Reports and the auditing process - 20102023.txt") as f:
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

      with open("Fact Sheet - Definitions of ‘Builder’ and ‘Building Work’.txt") as f:
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

      with open("Fact Sheet - Federal Safety Officers (FSOs)_0.txt") as f:
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

      with open("Fact Sheet - How to Get Accredited and Stay Accredited - April 2023.txt") as f:
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

      with open("Fact Sheet - International Companies and the Accreditation Process.txt") as f:
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

      with open("Fact Sheet - Joint Venture Arrangements_0.txt") as f:
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

      with open("Fact Sheet - Scheme Rule Changes.txt") as f:
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

      with open("Fact Sheet – Contracting for Australian Government funded building work.txt") as f:
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

      with open("Fact Sheet – Federal Safety Commissioner.txt") as f:
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

      with open("Fact Sheet – Hazards Research Project_0.txt") as f:
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

      with open("Fact Sheet – Powers of the Federal Safety Officers_0.txt") as f:
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

      with open("Fact Sheet – Safe Work Method Statements (SWMS).txt") as f:
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

      with open("Fact Sheet – Small builders and the Scheme_0.txt") as f:
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

      with open("Fact Sheet Verification of Competency - Mobile Plant_0.txt") as f:
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

'''
if st.button("Submit to AI", type="primary"):
     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=100,
     )
     augment_query = str(query_results["documents"])
    
     client_AI = OpenAI(api_key=apikey)
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
'''
st.write(len(collection.get()['documents']))
