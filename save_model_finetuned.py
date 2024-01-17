# Loading Model

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = 'tiiuae/falcon-40b'
# model_name = 'mistralai/Mistral-7B-v0.1'
# model_name = 'mistralai/Mixtral-8x7B-v0.1'

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
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

from peft import get_peft_model, LoraConfig, TaskType

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     target_modules = modules,
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1
# )

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# Training
CUTOFF_LEN = 512

from datasets import load_dataset

# dataset = load_dataset("json", data_files="cabrita-dataset-52k.json")
dataset = load_dataset("json", data_files="custom_dataset.json")
# dataset = load_dataset("json", data_files="custom_dataset_with_context.json")
# dataset = load_dataset("json", data_files="dataset_inputs.json")

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

tokenized_datasets = dataset.map(
    generate_and_tokenize_prompt,
    batched=False,
    num_proc=4,
    remove_columns=['instruction', 'input', 'output'],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, DataCollatorForSeq2Seq, set_seed

set_seed(42)

## EPOCHS = 3
# EPOCHS = 1
# GRADIENT_ACCUMULATION_STEPS = 1
# MICRO_BATCH_SIZE = 8
# LEARNING_RATE = 2e-4
# WARMUP_STEPS = 100
#### falcon_refined1 #####
# EPOCHS = 3
# # EPOCHS = 1
# GRADIENT_ACCUMULATION_STEPS = 2
# MICRO_BATCH_SIZE = 4 
# LEARNING_RATE = 1e-4
# WARMUP_STEPS = 500
#### falcon_refined1 #####
EPOCHS = 10
# EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 2
MICRO_BATCH_SIZE = 4 
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500

trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    args=Seq2SeqTrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optim = "paged_adamw_32bit",
        logging_steps=200,
        output_dir="qlora-cabrita",
        save_total_limit=3,
        gradient_checkpointing=True,
        # generation_config = GenerationConfig(temperature=0)
        generation_config = GenerationConfig(temperature=0.7)
    )
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("models/falcon_refined2")