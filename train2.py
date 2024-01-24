# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_dataset, Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# CUTOFF_LEN = 512

# model_name = 'tiiuae/falcon-40b'

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True
# )

# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map = "auto",
#                                              quantization_config=nf4_config)
#                                             #  trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

# def generate_prompt(instruction, input, output=None):
#   if input:
#     prompt = f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.
# ### Instrução:
# {instruction}
# ### Entrada:
# {input}
# ### Resposta:
# """
#   else:
#     prompt = f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.
# ### Instrução:
# {instruction}
# ### Resposta:
# """
#   if output:
#     prompt = f"{prompt}{output}"

#   return prompt


# def tokenize(prompt, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=CUTOFF_LEN,
#         padding=False,
#         return_tensors=None,
#     )
#     if (
#         result["input_ids"][-1] != tokenizer.eos_token_id
#         and len(result["input_ids"]) < CUTOFF_LEN
#         and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)

#     result["labels"] = result["input_ids"].copy()

#     return result


# def generate_and_tokenize_prompt(data_point):

#     full_prompt = generate_prompt(
#         data_point["instruction"],
#         data_point["input"],
#         data_point["output"],
#     )
#     tokenized_full_prompt = tokenize(full_prompt)

#     user_prompt = generate_prompt(
#         data_point["instruction"], data_point["input"]
#     )
#     tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
#     user_prompt_len = len(tokenized_user_prompt["input_ids"])

#     tokenized_full_prompt["labels"] = [
#         -100
#     ] * user_prompt_len + tokenized_full_prompt["labels"][
#         user_prompt_len:
#     ]
#     return tokenized_full_prompt


# test_set = load_dataset("json", data_files="eval_dataset.json")  # Substitua "eval_dataset.json" pelo caminho real do seu conjunto de teste


# # Tokenize the test dataset
# tokenized_eval_dataset = test_set.map(
#     generate_and_tokenize_prompt,
#     batched=False,
#     num_proc=4,
#     remove_columns=['instruction', 'input', 'output'],
#     load_from_cache_file=True,
#     desc="Running tokenizer on evaluation dataset",
# )

# model.eval()

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from peft import PeftModel
# from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = 'tiiuae/falcon-40b'
fine_tuned = "/home/tacouto/chatbot/chatbot_langchain/models/dataset_40qa"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map = "auto",
                                            quantization_config=nf4_config)
                                            #  trust_remote_code=True)


model = PeftModel.from_pretrained(model, fine_tuned)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


# Carregar o conjunto de teste
test_set = load_dataset("json", data_files="eval_dataset.json")
from datasets import Dataset
print(test_set)
test_set = test_set['train']
# test_set = Dataset.from_dict(test_set)
# test_set = Dataset.from_dict(test_set['train'])

CUTOFF_LEN = 512

def generate_prompt(instruction, input, output=None):
  if input:
    prompt = f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.
### Instrução:
{instruction}
### Entrada:
{input}
### Resposta:
"""
  else:
    prompt = f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.
### Instrução:
{instruction}
### Resposta:
"""
  if output:
    prompt = f"{prompt}{output}"

  return prompt


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):

    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)

    user_prompt = generate_prompt(
        data_point["instruction"], data_point["input"]
    )
    tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]
    return tokenized_full_prompt

tokenized_eval_dataset = test_set.map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=1,
    remove_columns=['instruction', 'input', 'output'],
    load_from_cache_file=True,
    desc="Running tokenizer on evaluation dataset",
)
print(f"SEM NADA: {tokenized_eval_dataset}")
print(f"Colunas: {tokenized_eval_dataset.column_names}")
# Avaliar o modelo no conjunto de teste
max_sequence_length = max(len(seq) for seq in tokenized_eval_dataset["input_ids"])
input_ids = torch.nn.functional.pad(
    torch.tensor(tokenized_eval_dataset["input_ids"]),
    pad=(0, max_sequence_length - len(seq)),
    value=tokenizer.pad_token_id
)
attention_mask = torch.nn.functional.pad(
    torch.tensor(tokenized_eval_dataset["attention_mask"]),
    pad=(0, max_sequence_length - len(seq)),
    value=0  # 0 é o valor padrão para attention_mask
)

# Criando um DataLoader para iterar sobre os dados
data_loader = DataLoader(list(zip(input_ids, attention_mask)), batch_size=8)

# Avaliar o modelo no conjunto de teste
model.eval()

predictions = []
# Iterar sobre os batches do DataLoader
for batch in data_loader:
    # Extrair elementos relevantes do batch
    batch_input_ids, batch_attention_mask = batch
    
    # Chamar o modelo com os elementos extraídos
    with torch.no_grad():
        results = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

    # Obter as previsões e adicionar à lista
    batch_predictions = results.logits.argmax(dim=-1).numpy()
    predictions.extend(batch_predictions)

# Converter as listas para arrays numpy
predictions = np.array(predictions)
labels = tokenized_eval_dataset["labels"].numpy()

# Calcular métricas
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

# Imprimir as métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')