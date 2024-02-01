from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = 'tiiuae/falcon-40b'

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map = "auto",
                                             quantization_config=nf4_config)
                                            #  trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


model.num_parameters()


from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset_18qa_isq_desc.json")
dataset


train_data = dataset['train']
test_size = 0.2
train_data = train_data.train_test_split(test_size=test_size)
train_set, test_set = train_data['train'], train_data['test']


print(train_set)
print(test_set)


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
        padding=True,
        return_tensors=None
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
    tokenized_full_prompts = {'input_ids': [],
                            'attention_mask': [],
                            'labels': []}
    for i_prompt in range(len(data_point['input'])):
        full_prompt = generate_prompt(
            data_point["instruction"][i_prompt],
            data_point["input"][i_prompt],
            data_point["output"][i_prompt],
        )
        # input_ids, attention_mask, labels
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = generate_prompt(
            data_point["instruction"][i_prompt], data_point["input"][i_prompt]
        )
        
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        tokenized_full_prompts["input_ids"].append(tokenized_full_prompt["input_ids"])
        tokenized_full_prompts["attention_mask"].append(tokenized_full_prompt["attention_mask"])
        tokenized_full_prompts["labels"].append(tokenized_full_prompt["labels"])

    return tokenized_full_prompts


tokenized_eval_dataset = test_set.map(
    generate_and_tokenize_prompt,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=['instruction', 'input', 'output'],
    # remove_columns=['instruction', 'input'],
    load_from_cache_file=False,
    desc="Running tokenizer on evaluation dataset",
)


tokenized_eval_dataset


tokenized_train_dataset = train_set.map(
    generate_and_tokenize_prompt,
    batched=True,
    batch_size=4,
    num_proc=4,
    remove_columns=['instruction', 'input', 'output'],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)


print(train_set[0])
print(tokenized_train_dataset)
print(tokenized_train_dataset[0]['input_ids'])
print(len(tokenized_train_dataset[0]['labels']))



from peft import get_peft_model, LoraConfig

peft_config = LoraConfig(
    # r=16,
    # lora_alpha=32,
    # target_modules=["query_key_value"],
    # lora_dropout=0.05,
    r=16,
    lora_alpha=64,
    target_modules=["query_key_value"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    print(pred)
    predictions = pred.predictions.argmax(axis=-1).flatten()
    labels = pred.label_ids.flatten()

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1":f1}


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

set_seed(42)
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2
MICRO_BATCH_SIZE = 8
LEARNING_RATE = 5e-5  
WARMUP_STEPS = 400 
trainer = Seq2SeqTrainer(
    model=model,
    compute_metrics=compute_metrics,
    # train_dataset=tokenized_datasets["train"],
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    args=Seq2SeqTrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps= 20,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optim = "paged_adamw_32bit",
        logging_steps=200,
        output_dir="qlora-cabrita",
        save_total_limit=3,
        gradient_checkpointing=True,
        generation_config = GenerationConfig(temperature=0)       
    )
)

trainer.model.eval()

with torch.no_grad():
    trainer.evaluate()


