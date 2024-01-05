import recordlinkage
import pandas as pd
from fuzzywuzzy import fuzz, process


def advanced_normalize(name, candidates):
    indexer = recordlinkage.Index()
    indexer.full()
    pairs = indexer.index(candidates)

    # Usar o processo de correspondência fuzzy para encontrar correspondências aproximadas
    matches = process.extract(name, candidates['name'], scorer=fuzz.token_set_ratio, limit=1)

    # Verificar se a correspondência atende a um limiar mínimo
    if matches and matches[0][1] > 80:  # Ajuste o limiar conforme necessário
        matched_name = matches[0][0]
        matched_candidate = candidates[candidates['name'] == matched_name].iloc[0]

        normalized_info = matched_candidate['name'], str(matched_candidate['phone']).replace(" ", ""), matched_candidate['email']
        return normalized_info
    else:
        return pd.Series({'name': 'empty', 'phone': 'empty', 'email': 'empty'})


# Carregar o arquivo Excel
excel_path = "/home/dev/chatbot_langchain-1/servicosISQ_tudo.xlsx"
df = pd.read_excel(excel_path)

# Nome para normalizar
name_to_normalize = 'André'

# Criar um DataFrame com candidatos
candidates = pd.DataFrame({
    'name': df['Responsável de serviço'],
    'phone': df['Telefone'],
    'email': df['Email de contacto']
})
candidates_no_duplicates = candidates.drop_duplicates(subset='name').reset_index(drop=True)

# Aplicar a função de normalização avançada
normalized_info = advanced_normalize(name_to_normalize,
                                    candidates_no_duplicates)

print(normalized_info)
