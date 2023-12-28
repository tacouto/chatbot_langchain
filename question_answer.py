import pandas as pd


def generate_questions(excel_path):
    '''Função que gera perguntas e respostas,
    com base no excel de serviços do ISQ'''

    df = pd.read_excel(excel_path)

    # Iterar sobre as linhas do Excel e gerar perguntas
    for index, row in df.iterrows():
        service = row['SERVIÇO']
        responsible = row['Responsável de serviço']
        phone = row['Telefone']
        email = row['Email de contacto']

        # Criar Diferentes perguntas
        question_1 = f"Who's responsible for the service {service}?"
        answer_1 = f"{responsible}, {phone}, {email}"
        yield question_1, answer_1

        question_2 = f"Does ISQ perform/do/have {service}?"
        answer_2 = f"Yes, contact {responsible}, {phone}, {email}"
        yield question_2, answer_2

        question_3 = f"Who should I contact for {service}?"
        answer_3 = f"Contact {responsible}, {phone}, {email}"
        yield question_3, answer_3

        question_4 = f"Tell me more about the {service} service."
        answer_4 = f"For more details, contact {responsible} at {phone} or {email}."
        yield question_4, answer_4

        question_5 = f"What are the contact details for {service}?"
        answer_5 = f"You can reach the {service} service at {phone} or {email}, and the responsible person is {responsible}."
        yield question_5, answer_5

        question_6 = f"Is the {service} service available?"
        answer_6 = f"Yes, the {service} service is currently available. For more information, contact {responsible} at {phone} or {email}."
        yield question_6, answer_6


excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"

# Gerar perguntas e respostas
for q, a in generate_questions(excel_path):
    print(f"Q: {q}\nA: {a}\n")
