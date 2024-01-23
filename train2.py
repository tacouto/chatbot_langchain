from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
CUTOFF_LEN = 512

# Carregando o modelo
model_name = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Adicionando as configurações para o treinamento com peft
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

modules = ['lm_head']

# Adicionando as configurações para o treinamento com peft
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=modules,
    r=16,
    lora_alpha=64,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# Configuração do treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir="./mistral-model",
    per_device_train_batch_size=4,
    save_total_limit=3,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    eval_steps=200,
    learning_rate=5e-5,
    num_train_epochs=5,
)

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

# Carregando e pré-processando o conjunto de dados
dataset = load_dataset("json", data_files="dataset_other.json")
train_data = dataset['train']

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

tokenized_train_data = train_data.map(generate_and_tokenize_prompt, batched=False)

# Configurando o treinador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    train_dataset=tokenized_train_data,
)

# Treinando o modelo
trainer.train()

# Avaliando o modelo no conjunto de teste
test_set = load_dataset("json", data_files="dataset_other.json")['test']
tokenized_test_data = test_set.map(generate_and_tokenize_prompt, batched=False)

results = trainer.predict(tokenized_test_data)
predictions = results.predictions.argmax(axis=-1).flatten()
labels = results.label_ids.flatten()

# Calculando métricas
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

# Imprimindo métricas
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
