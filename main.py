from langchain.document_loaders import TextLoader
import textwrap
import os
from langchain.chains.question_answering import load_qa_chain
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredExcelLoader
import csv
import pandas as pd

def xlsx_to_csv(xlsx_file, csv_file):
    # Carrega o arquivo Excel
    df = pd.read_excel(xlsx_file)

    # Salva o DataFrame como um arquivo CSV
    df.to_csv(csv_file, index=False)
    print(f"Conversion completed. Check '{csv_file_path}' for the result.")

import csv

def csv_to_txt(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            for row in csv_reader:
                service, service_responsible, contact, mail, dept_manager, general_number = row
                txt_file.write(f"User: Hi, how are you?\n")
                txt_file.write(f"Bot: I'm doing well, thank you! How can I assist you today?\n")
                txt_file.write(f"User: Can you tell me the contact of {service_responsible}?\n")
                txt_file.write(f"Bot: {service_responsible}'s contact number is '{contact}'.\n")
                txt_file.write(f"User: Can you tell me the responsible for the service {service}?\n")
                txt_file.write(f"Bot: The responsible for the service is {service_responsible}\n")
                txt_file.write(f"User: What is the email of {service_responsible}?\n")
                txt_file.write(f"Bot: The email is {mail}.\n")
                txt_file.write(f"User: Who is the department manager of the {service} service?\n")
                txt_file.write(f"Bot: The department manager is {dept_manager}.\n")
                txt_file.write(f"User: What is the general number?\n")
                txt_file.write(f"Bot: The general number is '{general_number}'.\n")
                txt_file.write(f"User: Thank you for the information!\n")
                txt_file.write(f"Bot: You're welcome! Feel free to ask.\n\n")

    print(f"Conversion completed. Check '{output_file}' for the result.")

def chat_bot(HUGGING_FACE_KEY, txt_file_path):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_KEY

    # Carregar o arquivo de texto
    loader = TextLoader(txt_file_path)
    document = loader.load()

    # # Preprocessamento
    # def wrap_text_preserve_newlines(text, width=110):
    #     lines = text.split('\n')
    #     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    #     wrapped_text = '\n'.join(wrapped_lines)
    #     return wrapped_text

    # Divisão do texto
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Treinamento do modelo de perguntas e respostas
    llm = langchain.llms.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
    # llm = langchain.llms.HuggingFaceHub(repo_id="facebook/bart-large", model_kwargs={"temperature": 0.8, "max_length": 512})
    chain = load_qa_chain(llm, chain_type="stuff")

    # Loop do chatbot
    print("Welcome to chatbot!")
    query = ""
    conversation_history = []

    while True:
        print("User:")
        query = input()

        if query == 'exit':
            print("Have a nice day!")
            break

        # Adicionar a nova entrada ao histórico da conversa
        conversation_history.append({"role": "user", "content": query})

        # Pesquisar documentos relevantes
        docs = db.similarity_search(query)

        # Executar o modelo de perguntas e respostas
        bot_response = chain.run(input_documents=docs, question=query)
        print(f"Chatbot: {bot_response}")


if __name__ == "__main__":

    data_xlsx_path = "/home/dev/git/chatbot/chatbot_langchain/servicosISQ_tudo.xlsx"
    csv_file_path = "/home/dev/git/chatbot/chatbot_langchain/servicosISQ_tudo.csv"
    xlsx_to_csv(data_xlsx_path, csv_file_path)

    txt_file = "/home/dev/git/chatbot/chatbot_langchain/data.txt"
    csv_to_txt(csv_file_path, txt_file)

    HUGGING_FACE_KEY = "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"
    chat_bot(HUGGING_FACE_KEY, txt_file)
