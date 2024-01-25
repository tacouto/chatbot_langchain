# from test import normalize_string
from normalize_string import normalize_string, advanced_normalize
from fuzzywuzzy import fuzz
import pandas as pd


def find_responsible(name, excel_path):
    '''Function to retrieve all services associated
    with a specified responsible person.'''

    df = pd.read_excel(excel_path)
    services_set = set()

    for index, row in df.iterrows():
        real_name = row['Responsável de serviço']
        normalized_name = normalize_string(real_name)
        similarity = fuzz.token_sort_ratio(name, normalized_name)

        # print(f"Name: {name}, Real Name: {real_name}, Similarity: {similarity}")

        # Find match
        if similarity > 45:  # value required to catch all luis ferreira
            services_set.add(row['SERVIÇO'])
            phone = row['Telefone']
            email = row['Email de contacto']

        # print(f"Services Set: {services_set}")

    if not services_set:
        return {'message': 'Person not found in ISQ file'}

    responsavel_info = {
        'Responsible': real_name,
        'Phone': phone,
        'Email': email,
        'Service(s)': ", ".join(services_set) if services_set else "empty"
    }

    return responsavel_info


# Exemple
excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"
responsable = 'joao'
invalid = 'Patricia'

result = find_responsible(responsable, excel_path)

if 'message' in result:
    print(result['message'])
else:
    print(f"Responsible: {result['Responsible']}, "
          f"Phone Contact: {result['Phone']}, "
          f"Email: {result['Email']}, "
          f"Service(s): {result['Service(s)']}")
