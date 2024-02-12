# import nltk
# from datasets import load_dataset
# import evaluate
# import numpy as np
# from transformers import AutoTokenizer, DataCollatorForSeq2Seq
# from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import torch
# torch.cuda.empty_cache()
# from peft import prepare_model_for_kbit_training
# from peft import get_peft_model, LoraConfig, TaskType
# from datasets import load_dataset
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

# model_name = 'tiiuae/falcon-7b'

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     # bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map = "auto",
#                                              quantization_config=nf4_config)
#                                             #  trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

# from peft import prepare_model_for_kbit_training

# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ## Find all linear layers to apply LORA, except those excluded by quantization and lm_head
# def find_all_linear_names(model):
#     import bitsandbytes as bnb

#     cls = bnb.nn.Linear4bit ## Fix as 4bits quantization
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])


#     if 'lm_head' in lora_module_names: # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# modules = find_all_linear_names(model)

# from peft import get_peft_model, LoraConfig, TaskType

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     target_modules = modules,
#     r=16,
#     lora_alpha=64,
#     # lora_dropout=0.1
#     lora_dropout=0.2
# )

# # peft_config = LoraConfig(
# #     # r=16,
# #     # lora_alpha=32,
# #     # target_modules=["query_key_value"],
# #     # lora_dropout=0.05,
# #     r=16,
# #     lora_alpha=64,
# #     target_modules=["query_key_value"],
# #     lora_dropout=0.1,
# #     # lora_dropout=0.2,
# #     bias="none",
# #     task_type="CAUSAL_LM"
# # )

# model = get_peft_model(model, peft_config)

# model.print_trainable_parameters()

# # Training
# # CUTOFF_LEN = 512
# CUTOFF_LEN = 128

# from datasets import load_dataset

# # dataset = load_dataset("json", data_files="datasets/custom_dataset.json")
# dataset = load_dataset("json", data_files="datasets/dataset2.json")

# from sklearn.model_selection import train_test_split
# train_data = dataset['train']
# # test_size = 0.1
# test_size = 0.2
# train_set, test_set = train_test_split(train_data, test_size=test_size, random_state=42)

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

# from datasets import Dataset

# train_set = Dataset.from_dict(train_set)
# tokenized_datasets = train_set.map(
#     generate_and_tokenize_prompt,
#     batched=False,  # Eu acho que é melhor colocar isto a TRUE...
#     num_proc=1,
#     remove_columns=['instruction', 'input', 'output'],
#     load_from_cache_file=True,
#     desc="Running tokenizer on dataset",
# )

# prefix = "summarize: "

# def preprocess_function(examples):
#     inputs = [prefix + doc for doc in examples["text"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

#     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# # tokenized_datasets = train_set.map(preprocess_function, batched=True)

# # Setup evaluation
# nltk.download("punkt", quiet=True)
# metric = evaluate.load("rouge")

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds

#     # decode preds and labels
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # rougeLSum expects newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     return result

# # Load pretrained model and evaluate model after each epoch
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# # training_args = Seq2SeqTrainingArguments(
# #     output_dir="./results",
# #     evaluation_strategy="epoch",
# #     learning_rate=2e-5,
# #     per_device_train_batch_size=16,
# #     per_device_eval_batch_size=4,
# #     weight_decay=0.01,
# #     save_total_limit=3,
# #     num_train_epochs=2,
# #     fp16=True,
# #     predict_with_generate=True
# # )

# test_set = Dataset.from_dict(test_set)
# # Tokenize o conjunto de dados de avaliação
# tokenized_eval_dataset = test_set.map(
#     generate_and_tokenize_prompt,
#     batched=False,
#     # num_proc=4,
#     num_proc=1,
#     remove_columns=['instruction', 'input', 'output'],
#     # remove_columns=['instruction', 'input'],
#     load_from_cache_file=True,
#     desc="Running tokenizer on evaluation dataset",
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     # train_dataset=tokenized_datasets["train"],
#     train_dataset=tokenized_datasets,
#     eval_dataset = tokenized_eval_dataset,
#     data_collator=data_collator,
#     args=Seq2SeqTrainingArguments(
#         per_device_eval_batch_size=1, # per_device_test_batch_size não existe
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         warmup_steps=400,
#         num_train_epochs=1,
#         learning_rate= 5e-5,
#         optim = "paged_adamw_32bit",
#         logging_steps=200,
#         output_dir="qlora-cabrita",
#         save_total_limit=2,
#         gradient_checkpointing=True,
#         generation_config = GenerationConfig(temperature=0)
#     ),
#     compute_metrics=compute_metrics
# )

# # trainer = Seq2SeqTrainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=tokenized_billsum["train"],
# #     eval_dataset=tokenized_billsum["test"],
# #     tokenizer=tokenizer,
# #     data_collator=data_collator,
# #     compute_metrics=compute_metrics
# # )


# # trainer.train()
# trainer.evaluate()

import nltk
from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split

dataset = load_dataset("json", data_files="datasets/dataset2.json")
train_data = dataset['train']
# test_size = 0.1
test_size = 0.2
train_set, test_set = train_test_split(train_data, test_size=test_size, random_state=42)


# Prepare and tokenize dataset
# billsum = load_dataset("billsum", split="ca_test").shuffle(seed=42).select(range(200))
# billsum = billsum.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("models/GPU_problem")
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = train_set.map(preprocess_function, batched=True)
tokenized_billsum = test_set.map(preprocess_function, batched=True)

# Setup evaluation
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

# Load pretrained model and evaluate model after each epoch
model = AutoModelForSeq2SeqLM.from_pretrained("models/GPU_problem")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    fp16=True,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
