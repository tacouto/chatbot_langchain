# Loading Model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
torch.cuda.empty_cache()
# model_name = 'tiiuae/falcon-40b'
model_name = 'mistralai/Mistral-7B-v0.1'
# model_name = 'tiiuae/falcon-7b'

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map = "auto",
                                             quantization_config=nf4_config)
                                            #  trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

## Find all linear layers to apply LORA, except those excluded by quantization and lm_head
def find_all_linear_names(model):
    import bitsandbytes as bnb

    cls = bnb.nn.Linear4bit ## Fix as 4bits quantization
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules = modules,
    r=16,
    lora_alpha=64,
    # lora_dropout=0.1
    lora_dropout=0.2
)

# peft_config = LoraConfig(
#     # r=16,
#     # lora_alpha=32,
#     # target_modules=["query_key_value"],
#     # lora_dropout=0.05,
#     r=16,
#     lora_alpha=64,
#     target_modules=["query_key_value"],
#     lora_dropout=0.1,
#     # lora_dropout=0.2,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# Training
# CUTOFF_LEN = 512
CUTOFF_LEN = 128

from datasets import load_dataset

# dataset = load_dataset("json", data_files="datasets/custom_dataset.json")
dataset = load_dataset("json", data_files="datasets/new2.json")

from sklearn.model_selection import train_test_split
train_data = dataset['train']
# test_size = 0.1
test_size = 0.2
train_set, test_set = train_test_split(train_data, test_size=test_size, random_state=42)

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

from datasets import Dataset

train_set = Dataset.from_dict(train_set)


print(train_set)

# tokenized_datasets = train_set.map(
#     generate_and_tokenize_prompt,
#     batched=False,
#     num_proc=4,
#     remove_columns=['instruction', 'input', 'output'],
#     load_from_cache_file=True,
#     desc="Running tokenizer on dataset",
# )

tokenized_datasets = train_set.map(
    generate_and_tokenize_prompt,
    batched=False,  # Eu acho que é melhor colocar isto a TRUE...
    num_proc=1,
    remove_columns=['instruction', 'input', 'output'],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

set_seed(42)

###### Ultimo modelo treinado ######
EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 1
MICRO_BATCH_SIZE = 4  # Experimentar com 64 ou 128
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500 

trainer = Seq2SeqTrainer(
    model=model,
    # train_dataset=tokenized_datasets["train"],
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    args=Seq2SeqTrainingArguments(
        per_device_eval_batch_size=1, # per_device_test_batch_size não existe
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optim = "paged_adamw_32bit",
        # optim = "adafactor",
        # optim="adamw_bnb_8bit",
        logging_steps=200,
        output_dir="qlora-cabrita",
        # save_total_limit=3,
        save_total_limit=2,
        gradient_checkpointing=True,
        # tf32=True,
        # fp16=True,
        # generation_config = GenerationConfig(temperature=0)
        generation_config = GenerationConfig(temperature=0.4)
    )
)

# default_args = {
#     "output_dir": "tmp",
#     "evaluation_strategy": "steps",
#     "num_train_epochs": 1,
#     "log_level": "error",
#     "report_to": "none",
# }

# training_args = Seq2SeqTrainingArguments(
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     fp16=True,
#     optim="adafactor",
#     **default_args,
# )

# trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tokenized_datasets)
# result = trainer.train()


model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("models/GPU_problem")

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# eval_dataset = load_dataset("json", data_files="eval_dataset.json")

test_set = Dataset.from_dict(test_set)
# Tokenize o conjunto de dados de avaliação
tokenized_eval_dataset = test_set.map(
    generate_and_tokenize_prompt,
    batched=False,
    # num_proc=4,
    num_proc=1,
    remove_columns=['instruction', 'input', 'output'],
    # remove_columns=['instruction', 'input'],
    load_from_cache_file=True,
    desc="Running tokenizer on evaluation dataset",
)
print(tokenized_eval_dataset.column_names)

results = trainer.predict(tokenized_eval_dataset)
print(f"Results: {results}\n\n")
predictions = results.predictions.argmax(axis=-1).flatten()  # Flatten para colocar numa unica dimensão.
predictions_norm = results.predictions.argmax(axis=-1)
print(f"Predictions (FLATTEN): {predictions}\n\n")
print(f"Predictions (Normal): {predictions_norm}\n\n")
labels = results.label_ids.flatten()
labels_norm = results.label_ids
print(f"Labels (FLATTEN): {labels}\n\n")
print(f"Labels (Normal): {labels_norm}\n\n")


accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
# test_results = trainer.predict(tokenized_eval_dataset)
# print(test_results)

# # Obtenha as previsões e os rótulos reais
# predictions = test_results.predictions.argmax(axis=-1)
# labels = test_results.label_ids

# import numpy as np

# # Achatando as previsões e rótulos
# flattened_predictions = predictions.ravel()
# flattened_labels = labels.ravel()

# # Calcule métricas do scikit-learn
# accuracy = accuracy_score(flattened_labels, flattened_predictions)
# precision = precision_score(flattened_labels, flattened_predictions, average="weighted")
# recall = recall_score(flattened_labels, flattened_predictions, average="weighted")
# f1 = f1_score(flattened_labels, flattened_predictions, average="weighted")

# # Imprima as métricas
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1-Score: {f1}')


# É NECESSÁRIO OS RESULTADOS QUE DEU PARA O FALCON (CURRENT_OUTPUT (ANTES ESTAVA A FICAR COM O OUTPUT (LINHA COMENTADA NO eval_dataset.map)))
# NECESSÁRIO SABER COMO FAZER AVALIACAO COM MUDANCA DE TREINO E TESTE 
# AUTOMATICAMENTE.. PARA JÁ DIVIDI O DATASET POR "MIM". ESTÁ 90% - 10% (TREINO - TESTE) APROXIMADAMENTE
####################################### MISTRAL ###########################
# Resuldos para epoch = 1
# GRADIENT_ACCUMULATION_STEPS = 2
# MICRO_BATCH_SIZE = 4 
# LEARNING_RATE = 1e-4
# WARMUP_STEPS = 500

# Accuracy: 0.015297202797202798
# Precision: 0.010709744013971689
# Recall: 0.015297202797202798
# F1-Score: 0.012107124537773444

# Resuldos para epoch = 3
# GRADIENT_ACCUMULATION_STEPS = 2
# MICRO_BATCH_SIZE = 4 
# LEARNING_RATE = 1e-4
# WARMUP_STEPS = 500

# Accuracy: 0.014714452214452214
# Precision: 0.011448462443271398
# Recall: 0.014714452214452214
# F1-Score: 0.0121659668391476


# Accuracy: 0.00802114243323442
# Precision: 0.0061052807913891454
# Recall: 0.00802114243323442
# F1-Score: 0.006794552616881799 # Estes ultimos valores foram um bocado à parte
####################################### MISTRAL ###########################


####################################### Falcon ###########################

# Resuldos para epoch = 10
# GRADIENT_ACCUMULATION_STEPS = 2
# MICRO_BATCH_SIZE = 4 
# LEARNING_RATE = 1e-5
# WARMUP_STEPS = 500
# Accuracy: 0.0
# Precision: 0.0
# Recall: 0.0
# F1-Score: 0.0


####################### KFOLD ####################
# from sklearn.model_selection import KFold
# import numpy as np
# # Defina o número de folds (k)
# num_folds = 3
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# # Converta seu conjunto de treinamento (tokenized_datasets["train"]) para uma lista de dicionários
# # train_dataset_list = tokenized_datasets["train"].to_dict("records")
# train_dataset_list = tokenized_datasets.to_dict("records")

# # Inicialize as métricas que você deseja avaliar ao longo dos folds
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []

# # Itere sobre os folds
# for fold, (train_index, val_index) in enumerate(kf.split(train_dataset_list)):
#     print(f"Fold {fold + 1}/{num_folds}")

#     # Divida o conjunto de treinamento em treinamento e validação
#     train_fold = [train_dataset_list[i] for i in train_index]
#     val_fold = [train_dataset_list[i] for i in val_index]

#     # Treine o modelo
#     trainer.train(train_fold)

#     # Avalie o modelo no conjunto de validação
#     test_results = trainer.predict(val_fold)

#     # Obtenha as previsões e rótulos reais
#     predictions = test_results.predictions.argmax(axis=-1)
#     labels = test_results.label_ids

#     # Achatando as previsões e rótulos
#     flattened_predictions = predictions.ravel()
#     flattened_labels = labels.ravel()

#     # Calcule métricas do scikit-learn
#     accuracy = accuracy_score(flattened_labels, flattened_predictions)
#     precision = precision_score(flattened_labels, flattened_predictions, average="weighted")
#     recall = recall_score(flattened_labels, flattened_predictions, average="weighted")
#     f1 = f1_score(flattened_labels, flattened_predictions, average="weighted")

#     # Armazene as métricas para este fold
#     accuracy_scores.append(accuracy)
#     precision_scores.append(precision)
#     recall_scores.append(recall)
#     f1_scores.append(f1)

# # Calcule as médias das métricas ao longo de todos os folds
# mean_accuracy = np.mean(accuracy_scores)
# mean_precision = np.mean(precision_scores)
# mean_recall = np.mean(recall_scores)
# mean_f1 = np.mean(f1_scores)

# # Imprima as médias das métricas
# print(f'Mean Accuracy: {mean_accuracy}')
# print(f'Mean Precision: {mean_precision}')
# print(f'Mean Recall: {mean_recall}')
# print(f'Mean F1-Score: {mean_f1}')


#############################################################
# Accuracy: 7.94533608771651e-05
# Precision: 8.52670214291528e-05
# Recall: 7.94533608771651e-05
# F1-Score: 8.225759714341797e-05
