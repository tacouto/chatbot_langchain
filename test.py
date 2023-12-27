from langchain.document_loaders import TextLoader
import textwrap
import os
from langchain.chains.question_answering import load_qa_chain
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import csv
import pandas as pd
import unicodedata
from io import StringIO
import math
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def xlsx_to_csv(xlsx_file, csv_file):
    '''
        This function converts the .xlsx file to a .csv file.
        Input1: .xlsx file path
        Input2: .csv file path (Path to save the .csv file)
        Output: .csv file store in path Input2
    '''
    df = pd.read_excel(xlsx_file)
    # Substituir valores nulos por uma string vazia
    df = df.fillna('empty')

    df.to_csv(csv_file, index=False)
    print(f"Conversion completed. Check '{csv_file_path}' for the result.")


def normalize_string(input_str):
    # Tentativa de "normalizar" os nomes. Luís -> Luis. João -> Joao. André -> Andre
    normalized_str = unicodedata.normalize('NFKD', input_str).encode('ascii', 'ignore').decode('utf-8') 
    return normalized_str


# def normalize_phone_number(phone_number):
#     '''
#         Contact numbers stay together.
#         Example:
#         Input -> 911 222 333
#         Output -> 911222333
#     '''
#     normalized_number = ''.join(char for char in phone_number if char.isdigit())
#     return normalized_number

def normalize_phone_number(number):
    '''
        Contact numbers stay together.
        Example:
        Input -> 911 222 333
        Output -> 911222333
    '''
    if str(number).startswith('9'):
        return int(number.replace(' ', ''))

    if str(number).startswith('2'):
        return int(round(float(number.replace(' ', ''))))

    if number == 'empty':
        return "empty"


def csv_to_txt(input_file, output_file):
    '''
        Converts the csv file to a txt file where multiple conversation are
        created for the differents services and responsibles, contacts, emails and department managers.

        Input1: .csv file path
        Input2: .txt file path (Path to save the .txt file)
        Output1: data.txt file store in path Input2
    '''
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            for row in csv_reader:
                service, service_responsible, contact, mail, dept_manager, general_number = row
                if service_responsible == 'empty' or dept_manager == 'empty' or mail == 'empty':  # Adaptar codigo para dept_manager e mail
                    txt_file.write(f"User: Hi, how are you?\n")
                    txt_file.write(f"Bot: Hi! How can I help you?\n")
                    # txt_file.write(f"User: ISQ presents any service of {normalize_string(service)}?\n")
                    # txt_file.write(f"Bot: Yes! The responsible is {normalize_string(service_responsible)}.\n")
                    txt_file.write(f"User: Can you tell me the contact for the {normalize_string(service)} responsible?\n")
                    txt_file.write(f"Bot: There is no responsible person although contact {normalize_phone_number(contact)}\n")
                    txt_file.write(f"User: Can you tell me the responsible for the service {normalize_string(service)}?\n")
                    txt_file.write(f"Bot: There is no responsible for the service {normalize_string(service)}\n")
                    txt_file.write(f"User: What is the email for {normalize_string(service)} service?\n")
                    txt_file.write(f"Bot: The email is {normalize_string(mail)}\n")
                    txt_file.write(f"User: Who is the department manager of the {normalize_string(service)} service?\n")
                    txt_file.write(f"Bot: The department manager of {normalize_string(service)} is {normalize_string(dept_manager)}\n")
                    txt_file.write(f"User: What is the general number?\n")
                    txt_file.write(f"Bot: The general number is {normalize_phone_number(contact)}.\n")
                    txt_file.write(f"User: What is the phone number for the {normalize_string(service)}?\n")
                    txt_file.write(f"Bot: There is no responsible. Contact {normalize_phone_number(contact)}\n")
                    txt_file.write(f"User: Thank you!\n")
                    txt_file.write(f"Bot: You're welcome!\n\n")
                else:
                    txt_file.write(f"User: Hi, how are you?\n")
                    txt_file.write(f"Bot: Hi! How can I help you?\n")
                    # txt_file.write(f"User: ISQ presents any service of {normalize_string(service)}?\n")
                    # txt_file.write(f"Bot: Yes! The responsible is {normalize_string(service_responsible)}.\n")
                    txt_file.write(f"User: Can you tell me the personal contact of {normalize_string(service_responsible)}?\n")
                    txt_file.write(f"Bot: {normalize_string(service_responsible)} personal contact number is {normalize_phone_number(contact)}.\n")
                    txt_file.write(f"User: Can you tell me the responsible for the service {normalize_string(service)}?\n")
                    txt_file.write(f"Bot: The responsible for the service is {normalize_string(service_responsible)}\n")
                    txt_file.write(f"User: What is the email of {normalize_string(service_responsible)}?\n")
                    txt_file.write(f"Bot: The email is {normalize_string(mail)}\n")
                    txt_file.write(f"User: Who is the department manager of the {normalize_string(service)} service?\n")
                    txt_file.write(f"Bot: The department manager of {normalize_string(service)} is {normalize_string(dept_manager)}.\n")
                    txt_file.write(f"User: What is the general number?\n")
                    txt_file.write(f"Bot: The general number is {normalize_phone_number(contact)}\n")
                    txt_file.write(f"User: What is the phone number of the {normalize_string(service_responsible)}?\n")
                    txt_file.write(f"Bot: The phone number of {normalize_string(service_responsible)}\n")
                    txt_file.write(f"User: Thank you!\n")
                    txt_file.write(f"Bot: You're welcome!\n\n")

    print(f"Conversion completed. Check '{output_file}' for the result.")

def chat_bot(HUGGING_FACE_KEY, txt_file_path):
    '''
        This function is where the chatbot is created, for that to happen is used LangChain.
        Is necessary a HUGGING FACE KEY.
        In the end of this function is create a while loop to maintain always the "conversation" 
        between user and bot.
        If user types "exit", the conversation is stopped.
        At this moment, the language between user and bot needs to be in English.
        Input1: HUGGING FACE KEY
        Input2: data.txt file path
        Input3: Chunk size appropriate to the problem
        output: Conversation between User and Bot
    '''
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_KEY

    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "Your name is Roboto, you are a nice virtual assistant. The context is:\n{context}"
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        "{question}"
    )
    # # Carregar o arquivo de texto
    # loader = TextLoader(txt_file_path)
    # document = loader.load()

    # # Preprocessamento
    # def wrap_text_preserve_newlines(text, width=110):
    #     lines = text.split('\n')
    #     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    #     wrapped_text = '\n'.join(wrapped_lines)
    #     return wrapped_text

    # Divisão do texto
    # text_splitter = CharacterTextSplitter(chunk_size = 0, chunk_overlap=0) # Serve para dividir o texto em documentos menores
    loader = TextLoader("./data.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    # document = TextLoader(txt_file).load()
    # docs = text_splitter.split_documents(documents) # Serve para dividir o document em documentos menores
    # docs = text_splitter.split_text(document)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()  # É responsável por gerar embeddings usando modelos pré-treinados do Hugging Face
    # db = FAISS.from_documents(docs, embeddings)  # É criado uma base de dados FAISS a partir dos embeddings dos documentos.
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # db = FAISS.from_texts(texts=docs, embedding=embeddings)
    # Embeddings -> Embeddings são representações numéricas de dados, como palavras, frases ou documentos inteiros, 
                #   que capturam informações semânticas e contextuais sobre esses dados.
    # FAISS -> A base de dados FAISS é uma estrutura eficiente para armazenar e pesquisar vetores de alta dimensionalidade.
    vectorstore = Chroma.from_documents(documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Treinar modelo de perguntas e respostas
    llm = langchain.llms.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
    # llm = langchain.llms.HuggingFaceHub(repo_id="facebook/bart-large", model_kwargs={"temperature": 0.8, "max_length": 512})~

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

    # chain = load_qa_chain(llm, chain_type="stuff")



    # É criado um objeto llm (Language Learning Model) usando o Hugging Face Hub
    # Os model_kwargs são argumentos adicionais que podem ser passados para o modelo, como a temperatura e o comprimento máximo das respostas geradas.
            #  A temperatura controla a aleatoriedade das respostas geradas. Se for 1.0, as respostas tendem a ser mais aleatorias.
                                                                        #    Se for 0.1, as respostas tendem a ser mais diretas.
    # Chama-se a função load_qa_chain para criar uma cadeia de perguntas e respostas.
    # O uso de "stuff" indica que a cadeia é usada para lidar com perguntas e respostas gerais, sem uma especificidade definida. Existem outras:
    # -> "tech": usado para perguntas e respostas relacionadas a tecnologia e ciência da computação.
    # -> "medical": usado para perguntas e respostas relacionadas a informações médicas e de saúde.
    # -> "legal": usado para perguntas e respostas relacionadas a questões legais e jurídicas.
    # -> "finance": usado para perguntas e respostas relacionadas a finanças e investimentos.

    # memory = ConversationBufferMemory()

    # Loop do chatbot
    print("Welcome to chatbot!")
    query = ""

    while True:
        print("User:")
        query = input()

        if query == 'exit':
            print("Have a nice day!")
            break

        # Adicionar a nova entrada ao histórico da conversa (Não funcina..)
        # conversation_history.append({"role": "user", "content": query})

        # # Pesquisar documentos relevantes
        # docs = db.similarity_search(query)

        # # Executar o modelo de perguntas e respostas
        # bot_response = chain.run(input_documents=docs, question=query)
        # print(f"Chatbot: {bot_response}")

        response = chain({'question': query})
        answer = response['answer']
        print(f"User: {query}")
        print(f"Bot: {answer}")


if __name__ == "__main__":

    data_xlsx_path = "/home/dev/git/chatbot/chatbot_langchain/servicosISQ_tudo.xlsx"
    # data_xlsx_path = '/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx'
    csv_file_path = "/home/dev/git/chatbot/chatbot_langchain/servicosISQ_tudo.csv"
    # csv_file_path = '/home/dev/chatbot_langchain-1/servicosISQ_tudo.csv'

    xlsx_to_csv(data_xlsx_path, csv_file_path)

    txt_file = "/home/dev/git/chatbot/chatbot_langchain/data.txt"
    # txt_file = '/home/dev/chatbot_langchain-1/data.txt'

    csv_to_txt(csv_file_path, txt_file)

    HUGGING_FACE_KEY = "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"
    chat_bot(HUGGING_FACE_KEY, txt_file)