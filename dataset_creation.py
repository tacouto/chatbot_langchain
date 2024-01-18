import pandas as pd
import json

def generate_questions(excel_path, output_file):
    df = pd.read_excel(excel_path)
    
    data = []
    for index, row in df.iterrows():
        service = row['SERVIÇO']
        responsible = row['Responsável de serviço']
        phone = row['Telefone']
        email = row['Email de contacto']

        question_1 = f"Who is the responsbile for the service {service}?"
        answer_1 = f"The responsible is {responsible}, {phone}, {email}"
        data.append({
            "instruction": f"'{question_1}'.",
            "input": f"\"I want the the responsible of the service {service} on ISQ\"",
            "output": f"\"{answer_1}\"",
        })

        question_2 = f"ISQ have/operates any {service} service?"
        answer_2 = f"Yes, contact {responsible}, {phone}, {email}"
        data.append({
            "instruction": f"'{question_2}'.",
            "input": f"\"Tell if ISQ provides this {service}\"",
            "output": f"\"{answer_2}\"",
        })

        question_3 = f"Who i need to contact for the service {service}?"
        answer_3 = f"Contact {responsible}, {phone}, {email}"
        data.append({
            "instruction": f"'{question_3}'.",
            "input": f"\"What is the contact number for the responsible of the {service} on ISQ\"",
            "output": f"\"{answer_3}\"",
        })

        question_4 = f"Tell me more about the service {service}."
        answer_4 = f"For more details, contact {responsible} with the contact number {phone} or mail {email}."
        data.append({
            "instruction": f"'{question_4}'.",
            "input": f"\"Give me more informations for the service {service} on ISQ\"",
            "output": f"\"{answer_4}\"",
        })

        question_5 = f"Tell me the contacts for the service {service}?"
        answer_5 = f"You can contact the {service} responsible by the phone number {phone} or email {email}, the responsible is {responsible}."
        data.append({
            "instruction": f"'{question_5}'.",
            "input": f"\"Provide me all the contacts for the service {service} on ISQ\"",
            "output": f"\"{answer_5}\"",
        }) 
        question_6 = f"The service {service} is available?"
        answer_6 = f"Yes, the service {service} is available. For more informations, contact the responsible {responsible} by the phone number {phone} or by email {email}."
        data.append({
            "instruction": f"'{question_6}'.",
            "input": f"\"This service {service}\"",
            "output": f"\"{answer_6}\"",
        })  
        # additional_info = f"O responsável pelo serviço {service} é {responsible} com o contacto {phone} ou {email}\n"
        # data.append({
        #     "instruction": "",
        #     "input": additional_info,
        #     "output": "",
        # })

    with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

if __name__ == "__main__":
    excel_path = "servicosISQ_tudo.xlsx"
    output_file = "custom_dataset_with_inputs.json"
    generate_questions(excel_path, output_file)
