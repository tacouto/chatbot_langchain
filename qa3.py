import pandas as pd
import random
import re
from datetime import datetime
from workalendar.europe import Portugal
import json
import sys

def fill_na(df):
    '''Fill all NAN in diferent columns'''
    # Substituir NaNs nas colunas relevantes
    df['SERVIÇO'].fillna('Unknown service', inplace=True)
    df['Responsável de serviço'].fillna('Unknown person', inplace=True)
    df['Telefone'].fillna('Unknown phone', inplace=True)
    df['Email de contacto'].fillna('Unknown email', inplace=True)

    return df


def dates_from_question(question):
    # Extrair datas no formato "MONTH DD" ou "DD/MM"
    date_pattern = re.compile(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}\b'
    )
    # Encontrar todas as correspondências na pergunta
    matches = date_pattern.findall(question)

    # Converter as datas em objetos datetime
    parsed_dates = []
    for match in matches:
        try:
            date_obj = datetime.strptime(match, "%B %d")  # "MONTH DD"
        except ValueError:
            date_obj = datetime.strptime(match, "%d/%m")  # Se falhar, "DD/MM"
        parsed_dates.append(date_obj)

    return parsed_dates


def is_weekend_holiday(date):
    '''Verifies if date is weekend or holiday in PT'''
    cal = Portugal()

    if datetime.now().weekday() >= 5 or not cal.is_working_day(date):
        today = datetime.now().date()
        if cal.is_working_day(today):
            return False
        else:
            return True


def provide_positive_feedback():
    '''Provides Positive Feedback'''
    positive_feedback = [
        "Great! Is there anything else I can help you with?",
        "Excellent! Do you have any more questions?",
        "Fantastic! Feel free to ask if you need further assistance.",
        "Awesome! Let me know if there's anything else you'd like to know.",
        "Wonderful! Is there any other information you're looking for?",
        "Perfect! If you have more inquiries, feel free to ask."
    ]
    return random.choice(positive_feedback)


def provide_negative_feedback():
    '''Provides Negative Feedback'''
    negative_feedback = [
        "I apologize if the information provided isn't meeting your expectations. Please let me know how I can assist you better.",
        "I'm sorry if that wasn't helpful. Is there a specific aspect you'd like more information on?",
        "I understand if the answer didn't fully address your question. Feel free to ask for more details or clarification.",
        "I appreciate your patience. If there's anything specific you're looking for, please guide me so I can assist you better.",
        "I'm here to help, and I'm sorry if the response didn't meet your needs. Are there other questions I can answer for you?",
        "Thank you for your consideration. If there's a different way I can assist you, please don't hesitate to ask."
    ]
    return random.choice(negative_feedback)


def generate_positive_answers(excel_path):
    '''Function that generates questions for positive answers based on the ISQ services Excel'''

    df = pd.read_excel(excel_path)
    df = fill_na(df)

    # Mensagem informativa sem link clicável
    # isq_website_url = "https://www.isq.pt/"
    # clickable_link = f"<a href='{isq_website_url}'>here</a>"
    info_message = "/nFor more details, you can visit our website."

    data = []
    # Iterate over Excel rows and generate questions
    for index, row in df.iterrows():
        service = row['SERVIÇO']
        responsible = row['Responsável de serviço']
        phone = row['Telefone']
        email = row['Email de contacto']

        # Create Different Positive questions
        question_1 = f"Who's responsible for the service {service}?"
        answer_1 = f"{responsible}, {phone}, {email}"
        intention_1 = "GetResponsibility"
        yield question_1, answer_1, intention_1
        #print(provide_positive_feedback())

        question_2 = f"Does ISQ perform/do/have {service}?"
        answer_2 = f"Certainly, you can reach out to {responsible}, {phone}, {email}.{info_message}"
        intention_2 = "ServiceAvailability"
        yield question_2, answer_2, intention_2
        #print(provide_positive_feedback())

        question_3 = f"Who should I contact for {service}?"
        answer_3 = f"Feel free to contact {responsible}, {phone}, {email}"
        intention_3 = "GetContactPerson"
        yield question_3, answer_3, intention_3
        #print(provide_positive_feedback())

        question_4 = f"Tell me more about the {service} service."
        answer_4 = f"For more details, contact {responsible} at {phone} or {email}."
        intention_4 = "GetServiceDetails"
        yield question_4, answer_4, intention_4
        #print(provide_positive_feedback())

        question_5 = f"What are the contact details for {service}?"
        answer_5 = f"You can reach the {service} service at {phone} or {email}, and the responsible person is {responsible}."
        intention_5 = "GetContactDetails"
        yield question_5, answer_5, intention_5
        #print(provide_positive_feedback())

        question_6 = f"Is the {service} service available?"
        answer_6 = f"Absolutely, the {service} service is currently available. For more information, contact {responsible} at {phone} or {email}."
        intention_6 = "ServiceAvailability"
        yield question_6, answer_6, intention_6
        #print(provide_positive_feedback())

        question_7 = f"How can I request service {service}?"
        answer_7 = f"You can request {service} service by contacting {responsible} through {phone} or {email}."
        intention_7 = "RequestService"
        yield question_7, answer_7, intention_7
        #print(provide_positive_feedback())

        question_8 = f"In addition to service {service}, what additional resources are available?"
        answer_8 = f"For more thorough information, please contact {responsible} through {phone} or {email}."
        intention_8 = "GetAdditionalResources"
        yield question_8, answer_8, intention_8
        #print(provide_positive_feedback())

        question_9 = f"Are there customization options for service {service}?"
        answer_9 = f"To answer that, you can get in touch with {responsible} by {phone} or {email}."
        intention_9 = "GetCustomizationOptions"
        yield question_9, answer_9, intention_9
        #print(provide_positive_feedback())


def generate_negative_answers(excel_path):
    '''Function that generates questions for negative answers based on the ISQ services Excel'''

    df = pd.read_excel(excel_path)
    df = fill_na(df)

    # Iterate over Excel rows and generate negative questions
    for index, row in df.iterrows():
        service = row['SERVIÇO']

        # Create Different negative questions
        question_1 = f"Is {service} available on weekends/holidays?"
        answer_1 = f"Unfortunately, {service} is not available on weekends nor holidays."
        intention_1 = "ServiceNotAvailableWeekendOrHoliday"
        #print(provide_negative_feedback())

        question_2 = f"Is {service} available 24/7?"
        answer_2 = f"Sadly, {service} is not available 24/7."
        intention_2 = "ServiceNotAvailable24/7"
        #print(provide_negative_feedback())

        question_3 = f"Is {service} free of charge?"
        answer_3 = f"Naturally, {service} is not free of charge."
        intention_3 = "ServiceNotFree"
        #print(provide_negative_feedback())

        question_4 = f"Is there a notification system for changes in {service}?"
        answer_4 = f"Sad to say, there is no notification system for changes in this service."
        intention_4 = "NotificationSystemNotAvailable"
        #print(provide_negative_feedback())

        question_5 = f"Is {service} only available for business customers?"
        answer_5 = f"As might be expected, {service} is not only available for business customers."
        intention_5 = "ServiceAvailableForEveryone"
        #print(provide_negative_feedback())

        # Verifica se a data introduzida pelo utilizador é weekend/holiday
        question_6 = f"Is {service} available on December 25?"
        dates = dates_from_question(question_6)

        # Lógica para verificar se o serviço não está disponível
        for date in dates:
            if is_weekend_holiday(date):
                # Resposta para fins de semana ou feriados
                answer_6 = f"Unfortunately, {service} is not available on weekends nor holidays."
                intention_6 = "ServiceNotAvailableWeekendOrHoliday"
                #print(provide_negative_feedback())

            else:
                # Resposta para dias de trabalho
                answer_6 = f"For more details, contact {row['Responsável de serviço']} at {row['Telefone']} or {row['Email de contacto']}."
                intention_6 = "GetServiceDetails"
                #print(provide_positive_feedback())

        yield question_1, answer_1, intention_1
        yield question_2, answer_2, intention_2
        yield question_3, answer_3, intention_3
        yield question_4, answer_4, intention_4
        yield question_5, answer_5, intention_5
        yield question_6, answer_6, intention_6


excel_path = "/home/tacouto/chatbot/chatbot_langchain/servicosISQ_tudo.xlsx"

# Specify the file path where you want to save the output in JSON format
output_json_path = "datasets/qa3.json"

data_output = []

# # Generate positive questions and answers with intentions
# for q, a, intent in generate_positive_answers(excel_path):
#     data_output.append({"Q": q, "A": a, "Intent": intent})

# # Generate negative questions and answers with intentions
# for q, a, intent in generate_negative_answers(excel_path):
#     data_output.append({"Q": q, "A": a, "Intent": intent})

# Generate positive questions and answers with intentions
for q, a, intent in generate_positive_answers(excel_path):
    data_output.append({"instruction": q, "input": "", "output": a})

# Generate negative questions and answers with intentions
for q, a, intent in generate_negative_answers(excel_path):
    data_output.append({"instruction": q, "input": "", "output": a})

df = pd.read_excel(excel_path)

data = []

question_1 = f"What is ISQ?"
answer_1 = f" refers to Instituto de Soldadura e Qualidade. The ISQ is a private, independent organization that provides services in areas such as inspection, testing, and certification of materials, products, and processes."
data_output.append({
    "instruction": f"'{question_1}'.",
    "input": "",
    "output": f"\"{answer_1}\"",
})

question_2 = "What does ISQ stand for?"
answer_2 = "ISQ stands for 'Instituto de Soldadura e Qualidade,' which translates to Institute of Welding and Quality. It is a private, independent organization in Portugal that offers services in inspection, testing, and certification of materials, products, and processes."
data_output.append({
    "instruction": f"'{question_2}'.",
    "input": "",
    "output": f"\"{answer_2}\"",
})

question_3 = "Can you explain the role of ISQ in Portugal?"
answer_3 = "ISQ plays a significant role in Portugal by providing services related to inspection, testing, and certification. Specializing in areas such as welding and quality control, the institute ensures the quality and safety of various industrial processes and products."
data_output.append({
    "instruction": f"'{question_3}'.",
    "input": "",
    "output": f"\"{answer_3}\"",
})

question_4 = "In which sectors does ISQ operate?"
answer_4 = "ISQ operates in various sectors, including but not limited to welding, non-destructive testing, and quality control. The institute's expertise extends to ensuring the quality and safety standards in different industrial processes and products."
data_output.append({
    "instruction": f"'{question_4}'.",
    "input": "",
    "output": f"\"{answer_4}\"",
})

question_5 = "What services does ISQ provide?"
answer_5 = "ISQ provides a range of services, including inspection, testing, and certification of materials, products, and processes. The institute's focus on quality assurance extends to various industrial applications, making it a crucial player in ensuring standards compliance."
data_output.append({
    "instruction": f"'{question_5}'.",
    "input": "",
    "output": f"\"{answer_5}\"",
})

service_arr = []
responsible_arr  = []
phone_arr  = []
email_arr  = []
depart_arr  = []
phone_general_arr  = [] 

for index, row in df.iterrows():
    # service = row['SERVIÇO']
    # responsible = row['Responsável de serviço']
    # phone = row['Telefone']
    # email = row['Email de contacto']
    # depart = row['Resp. de Departamento']
    # phone_general = row['Telefone geral']
    
    service = row['SERVIÇO'] if not pd.isna(row['SERVIÇO']) else 'Unknown'
    responsible = row['Responsável de serviço'] if not pd.isna(row['Responsável de serviço']) else 'Unknown'
    phone = row['Telefone'] if not pd.isna(row['Telefone']) else 'Unknown'
    email = row['Email de contacto'] if not pd.isna(row['Email de contacto']) else 'Unknown'
    depart = row['Resp. de Departamento'] if not pd.isna(row['Resp. de Departamento']) else 'Unknown'
    phone_general = row['Telefone geral'] if not pd.isna(row['Telefone geral']) else 'Unknown'

    service_arr.append(service)
    responsible_arr.append(responsible)
    phone_arr.append(phone)
    email_arr.append(email)
    depart_arr.append(depart)
    phone_general_arr.append(phone_general)        
    
    question_1 = f"Who is responsible for the {service} service?"
    answer_1 = f"The responsible party is {responsible}, reachable at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_1}'.",
        "input": "",
        "output": f"\"{answer_1}\"",
    })

    question_2 = f"Does ISQ offer any {service} service?"
    answer_2 = f"Yes, you can contact {responsible} at {phone} or {email} for more information."
    data_output.append({
        "instruction": f"'{question_2}'.",
        "input": "",
        "output": f"\"{answer_2}\"",
    })

    question_3 = f"How can I reach out to the service provider for {service}?"
    answer_3 = f"For {service}, please contact {responsible} at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_3}'.",
        "input": "",
        "output": f"\"{answer_3}\"",
    })

    question_4 = f"Tell me more details about the {service} service."
    answer_4 = f"For additional information on {service}, you can contact {responsible} at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_4}'.",
        "input": "",
        "output": f"\"{answer_4}\"",
    })

    question_5 = f"What are the contact details for the {service} service?"
    answer_5 = f"For {service}, contact {responsible} via phone ({phone}) or email ({email})."
    data_output.append({
        "instruction": f"'{question_5}'.",
        "input": "",
        "output": f"\"{answer_5}\"",
    }) 

    question_6 = f"Is the {service} service currently available?"
    answer_6 = f"Yes, the {service} service is available. Reach out to {responsible} at {phone} or {email} for details."
    data_output.append({
        "instruction": f"'{question_6}'.",
        "input": "",
        "output": f"\"{answer_6}\"",
    })

    question_7 = f"Who is in charge of the {service} department at ISQ?"
    answer_7 = f"{responsible} is in charge of the {service} department and can be contacted at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_7}'.",
        "input": "",
        "output": f"\"{answer_7}\"",
    })

    question_8 = f"What is the general contact number for ISQ?"
    answer_8 = f"The general contact number for ISQ is {phone_general}. For {service}, contact {responsible} at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_8}'.",
        "input": "",
        "output": f"\"{answer_8}\"",
    })

    question_9 = f"What other services does ISQ offer?"
    answer_9 = f"In addition to {service}, we also offer other services. For more information, contact {responsible} at {phone} or {email}."
    data_output.append({
        "instruction": f"'{question_9}'.",
        "input": "",
        "output": f"\"{answer_9}\"",
    })

with open(output_json_path, 'w') as json_file:
    json.dump(data_output, json_file, indent=2)