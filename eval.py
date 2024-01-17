# from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
# from datasets import load_dataset
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# # Carregar o modelo treinado
# model = AutoModelForCausalLM.from_pretrained("models/falcon_refined2")  # Substitua pelo caminho real do seu modelo
# tokenizer = AutoTokenizer.from_pretrained("models/falcon_refined2")  # Substitua pelo caminho real do seu tokenizer

# # Carregar o conjunto de dados de validação
# validation_dataset = load_dataset("json", data_files="eval_dataset.json")
# import torch

# def estimate_loss(model, dataloader, device):
#     model.eval()
#     total_loss = 0.0
#     num_batches = len(dataloader)

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch["input_ids"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(inputs, labels=labels)
#             loss = outputs.loss.item()

#             total_loss += loss

#     average_loss = total_loss / num_batches
#     model.train()

#     return average_loss

# # Exemplo de uso
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# val_dataloader = torch.utils.data.DataLoader(tokenized_validation_datasets["validation"], batch_size=4, shuffle=True)

# average_val_loss = estimate_loss(model, val_dataloader, device)
# print(f"A perda média no conjunto de validação é: {average_val_loss}")
import sys
from lm_eval.evaluator import simple_evaluate
# from huggingface_hub import login
# login()
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
sys.path.append('./lm-evaluation-harness-PTBR')
result = simple_evaluate(
    model="hf-llama-causal",
    model_args="pretrained=openlm-research/open_llama_3b,low_cpu_mem_usage=True,dtype=float16",
    tasks=['faquad'],
    num_fewshot=4,
    batch_size=1,
    device="cuda:0",
    limit=None,
    bootstrap_iters=100000,
    description_dict={"faquad": "Forneça uma Resposta dado o Contexto."},
)

print(result)