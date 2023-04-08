# you need to pip install openai, chroma and langchain
import os

from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"

loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

llm_35Turbo = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
llm_davinci003 = OpenAI(model_name="text-davinci-003", temperature=0.0)

qaChain_35Turbo = RetrievalQA.from_chain_type(llm_35Turbo, chain_type="stuff", retriever=retriever)
qaChain_davinci003 = RetrievalQA.from_chain_type(llm_davinci003, chain_type="stuff", retriever=retriever)

qaWithSource_35Turbo = RetrievalQAWithSourcesChain.from_chain_type(llm_35Turbo, chain_type="stuff", retriever=retriever)
qaWithSource_davinci003 = RetrievalQAWithSourcesChain.from_chain_type(llm_davinci003, chain_type="stuff", retriever=retriever)

query = "should we reduce the corporate tax in America?"

res = qaChain_35Turbo.run(query)
print('res of qa_35Turbo = ', res)
res = qaChain_davinci003.run(query)
print('res of qa_davinci003 = ', res)

res = qaWithSource_35Turbo({"question": query}, return_only_outputs=False)
print('qaWithSource_35Turbo = ', res)
res = qaWithSource_davinci003({"question": query}, return_only_outputs=False)
print('qaWithSource_davinci003 = ', res)

