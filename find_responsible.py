from test import normalize_string
from fuzzywuzzy import fuzz
import pandas as pd


def find_responsible(name, excel_path):
    df = pd.read_excel(excel_path)
    services_set = set()
    max_similarity = 0
    best_match = None

    for index, row in df.iterrows():
        real_name = normalize_string(row['Responsável de serviço'])
        similarity = fuzz.token_sort_ratio(name, real_name)

        # Determinar uma correspondência
        if similarity > 80 and similarity > max_similarity:
            services_set.add(row['SERVIÇO'])
            phone = row['Telefone']
            email = row['Email de contacto']

    if not services_set:
        return {'message': 'Persone not found in ISQ file'}
    
    responsavel_info = {
        'Responsible': name,
        'Phone': phone,
        'Email': email,
        'Services': ", ".join(services_set) if services_set else "empty"
    }

    return responsavel_info


# Exemplo de uso
excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"
responsavel = "Luís Ferreira"
invalid = 'Patricia'
resultado = find_responsible(responsavel, excel_path)

if 'message' in resultado:
    print(resultado['message'])
else:
    print(f"Responsible: {resultado['Responsible']}, "
          f"Phone Contact: {resultado['Phone']}, "
          f"Email: {resultado['Email']}, "
          f"Service(s): {resultado['Services']}")
