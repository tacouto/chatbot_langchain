# from langchain.embeddings.openai import OpenAIEmbeddings4
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import langchain
import os


system_message_prompt = SystemMessagePromptTemplate.from_template(
    "Your name is Roboto, you are a nice virtual assistant. The context is:\n{context}"
)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "{question}"
)

loader = TextLoader("./data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
documents = text_splitter.split_documents(documents)


HUGGING_FACE_KEY = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_KEY
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_history = [
    ("my name is David, what's your name?", "Hello David! My name is Roboto. How can I assist you today?"),
]
llm = langchain.llms.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
    
# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    ),
    combine_docs_chain_kwargs={
        "prompt": ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt,
        ]),
    },
)

while True:
    print("User:")
    query = input()

    if query == 'exit':
        print("Have a nice day!")
        break

    result = chain({"question": query, "chat_history": chat_history})
    print(result["answer"])