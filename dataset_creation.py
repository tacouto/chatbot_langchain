import pandas as pd
import json
import numpy as np

def generate_questions(excel_path, output_file):
    df = pd.read_excel(excel_path)

    data = []
    for index, row in df.iterrows():
        service = row['SERVIÇO']
        responsible = row['Responsável de serviço']
        phone = row['Telefone']
        email = row['Email de contacto']
        depart = row['Resp. de Departamento']
        phone_general = row['Telefone geral']

        question_1 = f"Who is responsible for the {service} service?"
        answer_1 = f"The responsible party is {responsible}, reachable at {phone} or {email}."
        data.append({
            "instruction": f"'{question_1}'.",
            "input": f"\"Tell me the responsible party for the {service} service at ISQ.\"",
            "output": f"\"{answer_1}\"",
        })

        question_2 = f"Does ISQ offer any {service} service?"
        answer_2 = f"Yes, you can contact {responsible} at {phone} or {email} for more information."
        data.append({
            "instruction": f"'{question_2}'.",
            "input": f"\"Confirm if ISQ provides the {service} service.\"",
            "output": f"\"{answer_2}\"",
        })

        # question_3 = f"How can I reach out to the service provider for {service}?"
        # answer_3 = f"For {service}, please contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_3}'.",
        #     "input": f"\"What is the contact number for the responsible of the {service} at ISQ?\"",
        #     "output": f"\"{answer_3}\"",
        # })

        question_4 = f"Tell me more details about the {service} service."
        answer_4 = f"For additional information on {service}, you can contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_4}'.",
            "input": f"\"Give me more details about the {service} service at ISQ.\"",
            "output": f"\"{answer_4}\"",
        })

        question_5 = f"What are the contact details for the {service} service?"
        answer_5 = f"For {service}, contact {responsible} via phone ({phone}) or email ({email})."
        data.append({
            "instruction": f"'{question_5}'.",
            "input": f"\"Provide contact information for the {service} service at ISQ.\"",
            "output": f"\"{answer_5}\"",
        }) 

        # question_6 = f"Is the {service} service currently available?"
        # answer_6 = f"Yes, the {service} service is available. Reach out to {responsible} at {phone} or {email} for details."
        # data.append({
        #     "instruction": f"'{question_6}'.",
        #     "input": f"\"Is the {service} service currently offered by ISQ?\"",
        #     "output": f"\"{answer_6}\"",
        # })

        # question_7 = f"Who is in charge of the {service} department at ISQ?"
        # answer_7 = f"{responsible} is in charge of the {service} department and can be contacted at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_7}'.",
        #     "input": f"\"Tell me the head of the {service} department at ISQ.\"",
        #     "output": f"\"{answer_7}\"",
        # })

        question_8 = f"What is the general contact number for ISQ?"
        answer_8 = f"The general contact number for ISQ is {phone_general}. For {service}, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_8}'.",
            "input": f"\"Provide the general contact number for ISQ.\"",
            "output": f"\"{answer_8}\"",
        })

        # question_9 = f"What other services does ISQ offer?"
        # answer_9 = f"In addition to {service}, we also offer other services. For more information, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_9}'.",
        #     "input": f"\"List the other services offered by ISQ.\"",
        #     "output": f"\"{answer_9}\"",
        # })

        # question_10 = f"Is there a service similar to {service} at ISQ?"
        # answer_10 = f"Yes, we also have other services. For more details, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_10}'.",
        #     "input": f"\"Is there a service similar to {service} at ISQ?\"",
        #     "output": f"\"{answer_10}\"",
        # })

        

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

if __name__ == "__main__":
    excel_path = "servicosISQ_tudo.xlsx"
    output_file = "dataset_other.json"
    generate_questions(excel_path, output_file)
