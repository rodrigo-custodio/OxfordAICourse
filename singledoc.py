import os
import sys
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI

from secret_key import openapi_key

os.environ['OPENAI_API_KEY'] = openapi_key

# load the document as before
loader = PyPDFLoader('./docs/2023_GPT4All_Technical_Report.pdf')
documents = loader.load()

# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

# we create the RetrievalQA chain, passing in the vectorstore as our source of
# information. Behind the scenes, this will only retrieve the relevant
# data from the vectorstore, based on the semantic similiarity between
# the prompt and the stored information

retriever=vectordb.as_retriever(search_kwargs={'k': 6})
"""
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)
"""
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    chain_type="map_reduce",
    retriever=retriever
)

# we can now exectute queries againse our Q&A chain
"""
chat_history=[]
query = input("Prompt: ")
result = qa_chain({'question': query, 'chat_history': chat_history})
print(result['answer'])

chat_history=[(query,result["answer"])]
query = input("Prompt: ")
result = qa_chain({'question': query, 'chat_history': chat_history})
print(result['answer'])

"""
query = None
chat_history=[]
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = qa_chain({'question': query, 'chat_history': chat_history})
  print(result['answer'])
  chat_history.append((query, result['answer']))
  query = None
