from peft import PeftModel
# from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

import json
def save_to_json(conversations, json_path):
    structured_conversations = []

    for conversation in conversations:
        if conversation["instruction"].lower() != 'exit':
            structured_conversation = {
                "instruction": conversation["instruction"],
                "input": "",
                "output": conversation["output"]
            }
            structured_conversations.append(structured_conversation)

    with open(json_path, "w") as json_file:
        json.dump(structured_conversations, json_file, indent=2)

def chat_bot(model_name, fine_tuned):


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

    def generate_prompt(instruction, input=None):
        if input:
            return f"""Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.
            
    ### Instrução:
    {instruction}

    ### Entrada:
    {input}

    ### Resposta:"""
        else:
            return f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

    ### Instrução:
    {instruction}

    ### Resposta:"""
        
    generation_config = GenerationConfig(
    do_sample = True,  # Permite "usar" tokens aleatórios (TRUE, é melhor estar sempre a TRUE..) e não a sequencia completa (FALSE) (Embora meio que está a ser sempre completa)
    temperature=0.2,  # Valor de aleatoridade. Quanto mais alto (max = 1) mais aleatória é.
    top_p=0.7,  # Percentagem dos tokens mais provaveis.
    # num_beams=40,  # Um valor maior geralmente levará a uma geração mais focada e coerente, enquanto um valor menor pode levar a uma geração mais diversificada, mas potencialmente menos coerent
    num_beams=10,)

    def evaluate(instruction, input=None):
        # conversations.append(instruction)
        conversations.append({"instruction": "", "input": "", "output": ""})
        conversations[-1]["instruction"] = instruction
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens=256
            max_new_tokens=64
        )
        for s in generation_output.sequences:
            output = tokenizer.decode(s)
            print("Bot:", output.split("### Resposta:")[1].strip())
            # Guardar a interação
            bot_response = output.split("### Resposta:")[1].strip()
            # conversations.append(bot_response)
            conversations[-1]["output"] = bot_response
    while(1):
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            return conversations
            # break
        evaluate(user_input)

conversations = []
if __name__ == "__main__":

    model_name = 'tiiuae/falcon-40b'
    # model_name = 'tiiuae/falcon-7b'
    # fine_tuned = "models/dataset2_without_input_40b"
    # fine_tuned = "models/dataset2_without_input_40b_plusepochs"  # EN
    # fine_tuned = "models/pt_dataset2_without_input_40b_plusepochs"  # PT
    fine_tuned = "models/en_pt_dataset"  # EN e PT
    # fine_tuned = "models/pt_description_dataset"  # PT descriptions
    conversations = chat_bot(model_name, fine_tuned)
    chat_dataset_path = "chat_dataset.json"
    save_to_json(conversations, chat_dataset_path)
