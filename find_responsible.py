from test import normalize_string
from fuzzywuzzy import fuzz
import pandas as pd


def find_responsible(name, excel_path):
    '''Function to retrieve all services associated
    with a specified responsible person.'''

    df = pd.read_excel(excel_path)
    services_set = set()
    max_similarity = 0

    for index, row in df.iterrows():
        real_name = normalize_string(row['Responsável de serviço'])
        similarity = fuzz.token_sort_ratio(name, real_name)

        # Find match
        if similarity > 80 and similarity > max_similarity:
            services_set.add(row['SERVIÇO'])
            phone = row['Telefone']
            email = row['Email de contacto']

    if not services_set:
        return {'message': 'Person not found in ISQ file'}

    responsavel_info = {
        'Responsible': name,
        'Phone': phone,
        'Email': email,
        'Services': ", ".join(services_set) if services_set else "empty"
    }

    return responsavel_info


# Exemple
excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"
responsable = "Luís Ferreira"
invalid = 'Patricia'

result = find_responsible(responsable, excel_path)

if 'message' in result:
    print(result['message'])
else:
    print(f"Responsible: {result['Responsible']}, "
          f"Phone Contact: {result['Phone']}, "
          f"Email: {result['Email']}, "
          f"Service(s): {result['Services']}")
