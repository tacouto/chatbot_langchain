from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

# Loading Model
model_name = 'tiiuae/falcon-40b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Customize the model with quantization and LORA as needed (as in the provided code)

# Load dataset
dataset = load_dataset("json", data_files="train_dataset")

# Tokenize and preprocess the dataset
def tokenize_function(examples):
    # Implement the tokenize function based on your dataset structure
    # Make sure to handle instruction, input, and output columns properly
    return tokenized_example

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['instruction', 'input', 'output'],
)

# Training Configuration
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    num_train_epochs=6,
    learning_rate=1.85e-4,
    output_dir="qlora-cabrita",
    save_total_limit=3,
    gradient_checkpointing=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    train_dataset=tokenized_datasets["train"],
)

# Training
trainer.train()

# Save the trained model
model.save_pretrained("models/falcon_refined3")

# # Optionally, you can evaluate the model on a test set
# dataset = load_dataset("json", data_files="eval_dataset")
# test_results = trainer.evaluate(tokenized_datasets["test"])
# print(test_results)

# Evaluation
eval_dataset = load_dataset("json", data_files="eval_dataset")
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['instruction', 'input', 'output'],
)

test_results = trainer.evaluate(tokenized_eval_dataset)
print(test_results)