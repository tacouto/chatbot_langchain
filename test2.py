from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = 'tiiuae/falcon-40b'
# model_name = 'tiiuae/falcon-7b'

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

model.num_parameters()


from datasets import load_dataset

# dataset = load_dataset("json", data_files="datasets/pt_dataset2_without_input.json")
# dataset = load_dataset("json", data_files="datasets/en_description_dataset.json")
# dataset = load_dataset("json", data_files="en_pt_dataset.json")

dataset = load_dataset("json", data_files="en_description_dataset.json")  # errfile2 e current_output_secondary.
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
        # print(f"tokenized_user_prompt: {tokenized_user_prompt}")
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]

        tokenized_full_prompts["input_ids"].append(tokenized_full_prompt["input_ids"])
        tokenized_full_prompts["attention_mask"].append(tokenized_full_prompt["attention_mask"])
        tokenized_full_prompts["labels"].append(tokenized_full_prompt["labels"])
    print (f"tokenized_full_prompts{tokenized_full_prompts}")
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


print(f"train_set[0]: {train_set[0]}")
print(f"tokenized_train_dataset: {tokenized_train_dataset}")
print(f"tokenized_train_dataset[0]['input_ids']: {tokenized_train_dataset[0]['input_ids']}")
print(f"len(tokenized_train_dataset[0]['labels']): {len(tokenized_train_dataset[0]['labels'])}")
model.enable_input_require_grads()

from peft import get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules = modules,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import corpus_bleu
def compute_metrics(pred):
    print(f"pred: {pred}")
    # predictions = pred.predictions.argmax(axis=-1).flatten()
    # labels = tokenized_train_dataset[0]['input_ids']
    # predictions = pred.predictions.argmax(dim=-1)
    predictions = torch.argmax(torch.from_numpy(pred.predictions), dim=-1)
    labels = pred.label_ids

    # Lista para armazenar rótulos e previsões não preenchidas (-100)
    filtered_labels = []
    filtered_predictions = []

    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if labels[i][j] != -100:
                prediction_value = predictions[i][j-1].item() 
                label_value = labels[i][j].item()

                print(f"Previsão: {prediction_value}, Rótulo Real: {label_value}")

                filtered_labels.append(label_value)
                filtered_predictions.append(prediction_value)

    accuracy = accuracy_score(filtered_labels, filtered_predictions)
    precision = precision_score(filtered_labels, filtered_predictions, average='weighted')
    recall = recall_score(filtered_labels, filtered_predictions, average='weighted')
    f1 = f1_score(filtered_labels, filtered_predictions, average='weighted')

    # Calcule a métrica BLEU
    references = [[str(label)] for label in filtered_labels]
    hypotheses = [str(prediction) for prediction in filtered_predictions]
    bleu_score = corpus_bleu(references, hypotheses)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'BLEU Score: {bleu_score}')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "bleu_score": bleu_score}


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

set_seed(42)
EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 2
MICRO_BATCH_SIZE = 8
# LEARNING_RATE = 5e-5  
LEARNING_RATE = 2e-4
WARMUP_STEPS = 400 
trainer = Seq2SeqTrainer(
    model=model,
    # train_dataset=tokenized_datasets["train"],
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    compute_metrics=compute_metrics,
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

model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

# model.save_pretrained("models/en_pt_dataset")
model.save_pretrained("models/en_description_dataset")

# trainer.model.eval()

trainer.evaluate()
