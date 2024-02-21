from peft import PeftModel
# from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# tokenizer = LLaMATokenizer.from_pretrained("luodian/llama-7b-hf")
# model = LLaMAForCausalLM.from_pretrained(
#     # "decapoda-research/llama-7b-hf",
#     "luodian/llama-7b-hf",
#     load_in_8bit=True,
#     device_map="auto",
# )

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
    num_beams=40,  # Um valor maior geralmente levará a uma geração mais focada e coerente, enquanto um valor menor pode levar a uma geração mais diversificada, mas potencialmente menos coerent
    )

    def evaluate(instruction, input=None):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256
        )
        for s in generation_output.sequences:
            output = tokenizer.decode(s)
            print("Bot:", output.split("### Resposta:")[1].strip())

    while(1):
        evaluate(input("User: "))
        if input == 'exit':
            break

    # evaluate(input("Instrução: "))
    # evaluate("Tell me all the services that ISQ have for calibrations?")

if __name__ == "__main__":

    model_name = 'tiiuae/falcon-40b'
    # model_name = 'tiiuae/falcon-7b'
    # fine_tuned = "models/dataset2_without_input_40b"
    fine_tuned = "models/dataset2_without_input_40b_plusepochs"  # MELHOR PARA JÁ!!!!! 
    chat_bot(model_name, fine_tuned)