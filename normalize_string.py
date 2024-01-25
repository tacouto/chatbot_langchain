import recordlinkage
import pandas as pd
from fuzzywuzzy import fuzz, process
from unidecode import unidecode


def normalize_string(s):
    return unidecode(str(s).lower())


def advanced_normalize(name, candidates):
    indexer = recordlinkage.Index()
    indexer.full()
    # pairs = indexer.index(candidates)

    # Normalizar o nome a ser procurado
    normalized_name = normalize_string(name)

    # Normalizar os nomes dos candidatos
    candidates['normalized_name'] = candidates['name'].apply(normalize_string)

    # Processo de correspondência fuzzy para encontrar aproximações
    matches = process.extract(normalized_name, candidates['normalized_name'],
                              scorer=fuzz.token_set_ratio,
                              limit=1)

    # Verificar se a correspondência atende a um limiar mínimo
    if matches and matches[0][1] > 80:
        matched_name = matches[0][0]

        matched_candidate = candidates[candidates['normalized_name'] == matched_name].iloc[0]  # noqa E541

        normalized_info = matched_candidate['name'], str(matched_candidate['phone']).replace(" ", ""), matched_candidate['email']  # noqa E541
        return normalized_info

    else:
        return pd.Series({'name': 'empty', 'phone': 'empty', 'email': 'empty'})


# Carregar o arquivo Excel
excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"
df = pd.read_excel(excel_path)

# Nome para normalizar
name_to_normalize = 'tania'

# Criar um DataFrame com candidatos
candidates = pd.DataFrame({
    'name': df['Responsável de serviço'],
    'phone': df['Telefone'],
    'email': df['Email de contacto']
})

# candidates_no_duplicates = candidates.drop_duplicates(subset='name').reset_index(drop=True)

# Aplicar a função de normalização avançada
normalized_info = advanced_normalize(name_to_normalize,
                                     candidates)

print(normalized_info)
