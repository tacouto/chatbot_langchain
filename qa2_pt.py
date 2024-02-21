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
        "Ótimo! Há mais alguma coisa com que eu possa ajudar?",
        "Excelente! Tem mais alguma pergunta?",
        "Fantástico! Sinta-se à vontade para perguntar se precisar de mais alguma assistência.",
        "Incrível! Avise-me se houver mais alguma coisa que gostaria de saber.",
        "Maravilhoso! Existe alguma outra informação que está procurando?",
        "Perfeito! Se tiver mais dúvidas, sinta-se à vontade para perguntar."

    ]
    return random.choice(positive_feedback)


def provide_negative_feedback():
    '''Provides Negative Feedback'''
    negative_feedback = [
        "Peço desculpa se a informação fornecida não está a corresponder às suas expectativas. Por favor, informe-me como posso ajudar de forma mais eficaz.",
        "Lamento se não foi útil. Existe algum aspeto específico sobre o qual gostaria de obter mais informações?",
        "Compreendo se a resposta não abordou totalmente a sua questão. Sinta-se à vontade para pedir mais detalhes ou esclarecimentos.",
        "Agradeço a sua paciência. Se houver algo específico que procura, por favor, indique-me para que eu possa ajudar de forma mais eficiente.",
        "Estou aqui para ajudar, e lamento se a resposta não atendeu às suas necessidades. Há outras perguntas em que posso ajudar?",
        "Agradeço a sua compreensão. Se houver uma maneira diferente pela qual eu possa ajudar, não hesite em perguntar."
    ]
    return random.choice(negative_feedback)


def generate_positive_answers(excel_path):
    '''Function that generates questions for positive answers based on the ISQ services Excel'''

    df = pd.read_excel(excel_path)
    df = fill_na(df)

    # Mensagem informativa sem link clicável
    # isq_website_url = "https://www.isq.pt/"
    # clickable_link = f"<a href='{isq_website_url}'>here</a>"
    info_message = "/nPara mais detalhes pode consultar o nosso website."

    # Iterate over Excel rows and generate questions
    for index, row in df.iterrows():
        service = row['SERVIÇO']
        responsible = row['Responsável de serviço']
        if responsible in ['João Pombo', 'Maria Manuel Farinha', 'João Brás Luis', 'Sandra Isabel Fernandes', 'Paula Gorjão', 'Ana Cristina Gouveia', 'Rui Mendes', \
                           'Frazão Guerreiro', 'Alexandre Levy', 'Rui A. Louro', 'Carlos Martins', 'Pedro Pinto', 'Luís Ferreira', 'Maria João Franco', 'Januário da Torre', 'Tânia Farinha', \
                           'Elsa Maria Cantiga', 'André Ramalho', 'Cristina Leão', 'Rogério Magalhães', 'Liliana P. Silva', 'Sara Leonardo', 'Luis Conde Santos', \
                            'Vasco Mendes Pires', 'João Paulo Figueiredo', 'Elsa Cantiga', 'José Azevedo']:
            continue
        phone = row['Telefone']
        email = row['Email de contacto']

        # Create Different Positive questions
        question_1 = f"Quem é responsável pelo serviço {service}?"
        answer_1 = f"{responsible}, {phone}, {email}"
        intention_1 = "GetResponsibility"
        yield question_1, answer_1, intention_1

        question_2 = f"O ISQ realiza/tem o serviço {service}?"
        answer_2 = f"Certamente, pode entrar em contacto com {responsible}, {phone}, {email}.{info_message}"
        intention_2 = "ServiceAvailability"
        yield question_2, answer_2, intention_2

        question_3 = f"A quem devo contactar para o serviço {service}?"
        answer_3 = f"Sinta-se à vontade para contactar {responsible}, {phone}, {email}"
        intention_3 = "GetContactPerson"
        yield question_3, answer_3, intention_3

        question_4 = f"Conte-me mais sobre o serviço {service}."
        answer_4 = f"Para mais detalhes, contacte {responsible} através do telefone {phone} ou do email {email}."
        intention_4 = "GetServiceDetails"
        yield question_4, answer_4, intention_4

        question_5 = f"Quais são os detalhes de contacto para o serviço {service}?"
        answer_5 = f"Pode contactar o serviço {service} através do telefone {phone} ou do email {email}, e a pessoa responsável é {responsible}."
        intention_5 = "GetContactDetails"
        yield question_5, answer_5, intention_5

        question_6 = f"O serviço {service} está disponível?"
        answer_6 = f"Absolutamente, o serviço {service} está atualmente disponível. Para mais informações, contacte {responsible} através do telefone {phone} ou do email {email}."
        intention_6 = "ServiceAvailability"
        yield question_6, answer_6, intention_6

        question_7 = f"Como posso solicitar o serviço {service}?"
        answer_7 = f"Pode solicitar o serviço {service} entrando em contacto com {responsible} através do telefone {phone} ou do email {email}."
        intention_7 = "RequestService"
        yield question_7, answer_7, intention_7

        question_8 = f"Além do serviço {service}, que recursos adicionais estão disponíveis?"
        answer_8 = f"Para informações mais detalhadas, entre em contacto com {responsible} através do telefone {phone} ou do email {email}."
        intention_8 = "GetAdditionalResources"
        yield question_8, answer_8, intention_8

        question_9 = f"Há opções de personalização para o serviço {service}?"
        answer_9 = f"Para responder a isso, pode entrar em contacto com {responsible} através do telefone {phone} ou do email {email}."
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
        responsible = row['Responsável de serviço']
        if responsible in ['João Pombo', 'Maria Manuel Farinha', 'João Brás Luis', 'Sandra Isabel Fernandes', 'Paula Gorjão', 'Ana Cristina Gouveia', 'Rui Mendes', \
                           'Frazão Guerreiro', 'Alexandre Levy', 'Rui A. Louro', 'Carlos Martins', 'Pedro Pinto', 'Luís Ferreira', 'Maria João Franco', 'Januário da Torre', 'Tânia Farinha', \
                           'Elsa Maria Cantiga', 'André Ramalho', 'Cristina Leão', 'Rogério Magalhães', 'Liliana P. Silva', 'Sara Leonardo', 'Luis Conde Santos', \
                            'Vasco Mendes Pires', 'João Paulo Figueiredo', 'Elsa Cantiga', 'José Azevedo']:
            continue

        # Create Different negative questions
        question_1 = f"O {service} está disponível aos fins de semana/feriados?"
        answer_1 = f"Infelizmente, o {service} não está disponível aos fins de semana nem feriados."
        intention_1 = "ServiceNotAvailableWeekendOrHoliday"
        # print(provide_negative_feedback())

        question_2 = f"O {service} está disponível 24/7?"
        answer_2 = f"Infelizmente, o {service} não está disponível 24/7."
        intention_2 = "ServiceNotAvailable24/7"
        # print(provide_negative_feedback())

        question_3 = f"O {service} é gratuito?"
        answer_3 = f"Naturalmente, o {service} não é gratuito."
        intention_3 = "ServiceNotFree"
        # print(provide_negative_feedback())

        question_4 = f"Há um sistema de notificação para alterações no {service}?"
        answer_4 = f"Infelizmente, não há um sistema de notificação para alterações neste serviço."
        intention_4 = "NotificationSystemNotAvailable"
        # print(provide_negative_feedback())

        question_5 = f"O {service} está disponível apenas para clientes empresariais?"
        answer_5 = f"Como seria de esperar, o {service} não está disponível apenas para clientes empresariais."
        intention_5 = "ServiceAvailableForEveryone"
        # print(provide_negative_feedback())

        # Verifica se a data introduzida pelo utilizador é um fim de semana/feriado
        question_6 = f"O {service} está disponível no dia 25 de dezembro?"
        dates = dates_from_question(question_6)

        answer_6 = "" 
        intention_6 = ""
        # Lógica para verificar se o serviço não está disponível
        for date in dates:
            if is_weekend_holiday(date):
                # Resposta para fins de semana ou feriados
                answer_6 = f"Infelizmente, o {service} não está disponível aos fins de semana nem feriados."
                intention_6 = "ServiceNotAvailableWeekendOrHoliday"
                #print(provide_negative_feedback())

            else:
                # Resposta para dias de trabalho
                answer_6 = f"Para mais detalhes, entre em contato com {row['Responsável de serviço']} através do telefone {row['Telefone']} ou do email {row['Email de contacto']}."
                intention_6 = "GetServiceDetails"
                #print(provide_positive_feedback())

        yield question_1, answer_1, intention_1
        yield question_2, answer_2, intention_2
        yield question_3, answer_3, intention_3
        yield question_4, answer_4, intention_4
        yield question_5, answer_5, intention_5
        yield question_6, answer_6, intention_6


excel_path = "servicosISQ.xlsx"

# Specify the file path where you want to save the output in JSON format
output_json_path = "datasets/pt_dataset2_without_input.json"

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

services = []

for index, row in df.iterrows():
        # service = row['SERVIÇO']
        # responsible = row['Responsável de serviço']
        # phone = row['Telefone']
        # email = row['Email de contacto']
        # depart = row['Resp. de Departamento']
        # phone_general = row['Telefone geral']
        
        service = row['SERVIÇO'] if not pd.isna(row['SERVIÇO']) else 'Unknown'
        responsible = row['Responsável de serviço'] if not pd.isna(row['Responsável de serviço']) else 'Unknown'

        if responsible in ['João Pombo', 'Maria Manuel Farinha', 'João Brás Luis', 'Sandra Isabel Fernandes', 'Paula Gorjão', 'Ana Cristina Gouveia', 'Rui Mendes', \
                           'Frazão Guerreiro', 'Alexandre Levy', 'Rui A. Louro', 'Carlos Martins', 'Pedro Pinto', 'Luís Ferreira', 'Maria João Franco', 'Januário da Torre', 'Tânia Farinha', \
                           'Elsa Maria Cantiga', 'André Ramalho', 'Cristina Leão', 'Rogério Magalhães', 'Liliana P. Silva', 'Sara Leonardo', 'Luis Conde Santos', \
                            'Vasco Mendes Pires', 'João Paulo Figueiredo', 'Elsa Cantiga', 'José Azevedo']:
            continue
        
        services.append(service)
        phone = row['Telefone'] if not pd.isna(row['Telefone']) else 'Unknown'
        email = row['Email de contacto'] if not pd.isna(row['Email de contacto']) else 'Unknown'
        depart = row['Resp. de Departamento'] if not pd.isna(row['Resp. de Departamento']) else 'Unknown'
        phone_general = row['Telefone geral'] if not pd.isna(row['Telefone geral']) else 'Unknown'
        
        question_1 = f"Quem é responsável pelo serviço {service}?"
        answer_1 = f"A entidade responsável é {responsible}, contactável através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_1}'.",
            "input": "",
            "output": f"\"{answer_1}\"",
        })

        question_2 = f"O ISQ oferece algum serviço de {service}?"
        answer_2 = f"Sim, pode contactar {responsible} através do telefone {phone} ou do email {email} para obter mais informações."
        data_output.append({
            "instruction": f"'{question_2}'.",
            "input": "",
            "output": f"\"{answer_2}\"",
        })

        question_3 = f"Como posso contactar o fornecedor de serviços para {service}?"
        answer_3 = f"Para o serviço de {service}, por favor, entre em contacto com {responsible} através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_3}'.",
            "input": "",
            "output": f"\"{answer_3}\"",
        })

        question_4 = f"Dê-me mais detalhes sobre o serviço {service}."
        answer_4 = f"Para informações adicionais sobre o serviço {service}, pode contactar {responsible} através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_4}'.",
            "input": "",
            "output": f"\"{answer_4}\"",
        })

        question_5 = f"Quais são os detalhes de contacto para o serviço {service}?"
        answer_5 = f"Para o serviço de {service}, contacte {responsible} através do telefone ({phone}) ou do email ({email})."
        data_output.append({
            "instruction": f"'{question_5}'.",
            "input": "",
            "output": f"\"{answer_5}\"",
        })

        question_6 = f"O serviço {service} está atualmente disponível?"
        answer_6 = f"Sim, o serviço {service} está disponível. Entre em contacto com {responsible} através do telefone {phone} ou do email {email} para mais detalhes."
        data_output.append({
            "instruction": f"'{question_6}'.",
            "input": "",
            "output": f"\"{answer_6}\"",
        })

        question_7 = f"Quem está encarregue do departamento de {service} no ISQ?"
        answer_7 = f"{responsible} está encarregado do departamento de {service} e pode ser contactado através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_7}'.",
            "input": "",
            "output": f"\"{answer_7}\"",
        })

        question_8 = f"Qual é o número de contacto geral para o ISQ?"
        answer_8 = f"O número de contacto geral para o ISQ é {phone_general}. Para o serviço de {service}, contacte {responsible} através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_8}'.",
            "input": "",
            "output": f"\"{answer_8}\"",
        })

        question_9 = f"Que outros serviços oferece o ISQ?"
        answer_9 = f"Além do {service}, oferecemos também outros serviços. Para mais informações, contacte {responsible} através do telefone {phone} ou do email {email}."
        data_output.append({
            "instruction": f"'{question_9}'.",
            "input": "",
            "output": f"\"{answer_9}\"",
        })


question_1 = f"O que é o ISQ?"
answer_1 = f"ISQ refere-se ao Instituto de Soldadura e Qualidade. O ISQ é uma organização privada e independente que oferece serviços nas áreas de inspeção, teste e certificação de materiais, produtos e processos."
data_output.append({
    "instruction": f"'{question_1}'.",
    "input": "",
    "output": f"\"{answer_1}\"",
})

question = f"Quais as funções do ISQ?"
answer = f"ISQ remete a Instituto de Soldadura e Qualidade. O ISQ desempenha um papel importante através de serviços relacionados com a inspeção, ensaio e certificação. Especializado em áreas como a soldadura e o controlo de qualidade, o instituto garante a qualidade e segurança de diversos processos e produtos industriais."
data_output.append({
    "instruction": f"'{question}'.",
    "input": "",
    "output": f"\"{answer}\"",
})


question_2 = "O que significa ISQ?"
answer_2 = "ISQ significa 'Instituto de Soldadura e Qualidade', que se traduz para Instituto de Soldadura e Qualidade. É uma organização privada e independente em Portugal que oferece serviços de inspeção, teste e certificação de materiais, produtos e processos."
data_output.append({
    "instruction": f"'{question_2}'.",
    "input": "",
    "output": f"\"{answer_2}\"",
})

question_3 = "Pode explicar o papel do ISQ em Portugal?"
answer_3 = "O ISQ desempenha um papel significativo em Portugal, fornecendo serviços relacionados com inspeção, teste e certificação. Especializado em áreas como soldadura e controlo de qualidade, o instituto assegura a qualidade e segurança de vários processos industriais e produtos."
data_output.append({
    "instruction": f"'{question_3}'.",
    "input": "",
    "output": f"\"{answer_3}\"",
})

question_4 = "Em que setores opera o ISQ?"
answer_4 = "O ISQ opera em vários setores, incluindo, mas não se limitando a, soldadura, ensaios não destrutivos e controlo de qualidade. A expertise do instituto estende-se a assegurar os padrões de qualidade e segurança em diferentes processos industriais e produtos."
data_output.append({
    "instruction": f"'{question_4}'.",
    "input": "",
    "output": f"\"{answer_4}\"",
})

question_5 = "Que serviços oferece o ISQ?"
answer_5 = "O ISQ oferece uma variedade de serviços, incluindo inspeção, teste e certificação de materiais, produtos e processos. O foco do instituto na garantia de qualidade estende-se a várias aplicações industriais, tornando-o um interveniente crucial na conformidade com padrões."
data_output.append({
    "instruction": f"'{question_5}'.",
    "input": "",
    "output": f"\"{answer_5}\"",
})

print(services)
with open(output_json_path, 'w') as json_file:
    json.dump(data_output, json_file, indent=2)

# Remover os serviços com unknown:
    
def remover_conversas_desconhecidas(dados_json):
    conversas_filtradas = []

    for conversa in dados_json:
        if "Unknown" not in conversa["output"]:
            conversas_filtradas.append(conversa)

    return conversas_filtradas

# Ler o arquivo JSON
caminho_arquivo_json = "/home/tacouto/chatbot/chatbot_langchain/datasets/pt_dataset2_without_input.json"  # Substitua pelo caminho real do seu arquivo JSON
with open(caminho_arquivo_json, "r") as arquivo_json:
    dados_json = json.load(arquivo_json)

conversas_filtradas = remover_conversas_desconhecidas(dados_json)

# Escrever o novo arquivo JSON
novo_caminho_arquivo_json = "/home/tacouto/chatbot/chatbot_langchain/datasets/pt_dataset2_without_input.json"  # Substitua pelo caminho desejado para o novo arquivo JSON
with open(novo_caminho_arquivo_json, "w") as novo_arquivo_json:
    json.dump(conversas_filtradas, novo_arquivo_json, indent=2)
