import pandas as pd
import json
import numpy as np
from difflib import get_close_matches

# def encontrar_servicos_por_correspondencia(s, palavras_chave):
#     servicos_encontrados = []
#     for palavra_chave in palavras_chave.split():
#         matches = get_close_matches(palavra_chave, s, n=1, cutoff=0.6)
#         if matches:
#             servicos_encontrados.append(matches[0])
#     return servicos_encontrados

# def show_services(excel_path, output_file, s):
#     df = pd.read_excel(excel_path)

#     servicos_encontrados = encontrar_servicos_por_correspondencia(s, "Tell me all the services that ISQ provides for GPL")
    
#     if servicos_encontrados:
#         print("Serviços encontrados:")
#         for servico in servicos_encontrados:
#             print(servico)
#     else:
#         print("Nenhum serviço correspondente encontrado.")

def generate_questions(excel_path, output_file):
    df = pd.read_excel(excel_path)

    data = []

    question_1 = f"What is ISQ?"
    answer_1 = f" refers to Instituto de Soldadura e Qualidade. The ISQ is a private, independent organization that provides services in areas such as inspection, testing, and certification of materials, products, and processes."
    data.append({
        "instruction": f"'{question_1}'.",
        "input": f"\"Tell me the meaning of ISQ.\"",
        "output": f"\"{answer_1}\"",
    })

    question_2 = "What does ISQ stand for?"
    answer_2 = "ISQ stands for 'Instituto de Soldadura e Qualidade,' which translates to Institute of Welding and Quality. It is a private, independent organization in Portugal that offers services in inspection, testing, and certification of materials, products, and processes."
    data.append({
        "instruction": f"'{question_2}'.",
        "input": "\"Provide the acronym ISQ and its full form.\"",
        "output": f"\"{answer_2}\"",
    })

    question_3 = "Can you explain the role of ISQ in Portugal?"
    answer_3 = "ISQ plays a significant role in Portugal by providing services related to inspection, testing, and certification. Specializing in areas such as welding and quality control, the institute ensures the quality and safety of various industrial processes and products."
    data.append({
        "instruction": f"'{question_3}'.",
        "input": "\"What is the role of ISQ in Portugal?\"",
        "output": f"\"{answer_3}\"",
    })

    question_4 = "In which sectors does ISQ operate?"
    answer_4 = "ISQ operates in various sectors, including but not limited to welding, non-destructive testing, and quality control. The institute's expertise extends to ensuring the quality and safety standards in different industrial processes and products."
    data.append({
        "instruction": f"'{question_4}'.",
        "input": "\"Tell me about the sectors in which ISQ operates.\"",
        "output": f"\"{answer_4}\"",
    })

    question_5 = "What services does ISQ provide?"
    answer_5 = "ISQ provides a range of services, including inspection, testing, and certification of materials, products, and processes. The institute's focus on quality assurance extends to various industrial applications, making it a crucial player in ensuring standards compliance."
    data.append({
        "instruction": f"'{question_5}'.",
        "input": "\"Describe the services offered by ISQ.\"",
        "output": f"\"{answer_5}\"",
    })
    
    service_arr = []
    responsible_arr  = []
    phone_arr  = []
    email_arr  = []
    depart_arr  = []
    phone_general_arr  = [] 

    for index, row in df.iterrows():
        # service = row['SERVIÇO']
        # responsible = row['Responsável de serviço']
        # phone = row['Telefone']
        # email = row['Email de contacto']
        # depart = row['Resp. de Departamento']
        # phone_general = row['Telefone geral']
        
        service = row['SERVIÇO'] if not pd.isna(row['SERVIÇO']) else 'Unknown'
        responsible = row['Responsável de serviço'] if not pd.isna(row['Responsável de serviço']) else 'Unknown'

        phone = row['Telefone'] if not pd.isna(row['Telefone']) else 'Unknown'
        email = row['Email de contacto'] if not pd.isna(row['Email de contacto']) else 'Unknown'
        depart = row['Resp. de Departamento'] if not pd.isna(row['Resp. de Departamento']) else 'Unknown'
        phone_general = row['Telefone geral'] if not pd.isna(row['Telefone geral']) else 'Unknown'

        service_arr.append(service)
        responsible_arr.append(responsible)
        phone_arr.append(phone)
        email_arr.append(email)
        depart_arr.append(depart)
        phone_general_arr.append(phone_general)        
        
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

        question_3 = f"How can I reach out to the service provider for {service}?"
        answer_3 = f"For {service}, please contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_3}'.",
            "input": f"\"What is the contact number for the responsible of the {service} at ISQ?\"",
            "output": f"\"{answer_3}\"",
        })

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

        question_6 = f"Is the {service} service currently available?"
        answer_6 = f"Yes, the {service} service is available. Reach out to {responsible} at {phone} or {email} for details."
        data.append({
            "instruction": f"'{question_6}'.",
            "input": f"\"Is the {service} service currently offered by ISQ?\"",
            "output": f"\"{answer_6}\"",
        })

        question_7 = f"Who is in charge of the {service} department at ISQ?"
        answer_7 = f"{responsible} is in charge of the {service} department and can be contacted at {phone} or {email}."
        data.append({
            "instruction": f"'{question_7}'.",
            "input": f"\"Tell me the head of the {service} department at ISQ.\"",
            "output": f"\"{answer_7}\"",
        })

        question_8 = f"What is the general contact number for ISQ?"
        answer_8 = f"The general contact number for ISQ is {phone_general}. For {service}, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_8}'.",
            "input": f"\"Provide the general contact number for ISQ.\"",
            "output": f"\"{answer_8}\"",
        })

        question_9 = f"What other services does ISQ offer?"
        answer_9 = f"In addition to {service}, we also offer other services. For more information, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_9}'.",
            "input": f"\"List the other services offered by ISQ.\"",
            "output": f"\"{answer_9}\"",
        })

        question_10 = f"Is there a service similar to {service} at ISQ?"
        answer_10 = f"Yes, we also have other services. For more details, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_10}'.",
            "input": f"\"Is there a service similar to {service} at ISQ?\"",
            "output": f"\"{answer_10}\"",
        })

        question_11 = f"What are the working hours for {service} at ISQ?"
        answer_11 = f"The working hours for {service} are [insert working hours here]. You can contact {responsible} at {phone} or {email} during these hours."
        data.append({
            "instruction": f"'{question_11}'.",
            "input": f"\"What are the working hours for the {service} service at ISQ?\"",
            "output": f"\"{answer_11}\"",
        })

        question_12 = f"Can I schedule an appointment for the {service} service?"
        answer_12 = f"Yes, you can schedule an appointment for {service} by reaching out to {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_12}'.",
            "input": f"\"How can I schedule an appointment for the {service} service at ISQ?\"",
            "output": f"\"{answer_12}\"",
        })

        question_13 = f"Are there any fees associated with the {service} service at ISQ?"
        answer_13 = f"There may be fees associated with {service}. For detailed information on fees, please contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_13}'.",
            "input": f"\"Are there any fees for the {service} service at ISQ?\"",
            "output": f"\"{answer_13}\"",
        })

        question_14 = f"Is there an online platform for accessing {service} information?"
        answer_14 = f"Yes, you can access information about {service} on our online platform. For any specific queries, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_14}'.",
            "input": f"\"Is there an online platform for {service} information at ISQ?\"",
            "output": f"\"{answer_14}\"",
        })

        question_15 = f"Can I provide feedback about the {service} service at ISQ?"
        answer_15 = f"Yes, we welcome feedback. Feel free to provide your feedback on the {service} service by contacting {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_15}'.",
            "input": f"\"How can I provide feedback about the {service} service at ISQ?\"",
            "output": f"\"{answer_15}\"",
        })

        question_16 = f"Is there any training available for using {service} at ISQ?"
        answer_16 = f"Yes, we offer training for using {service}. Contact {responsible} at {phone} or {email} to inquire about available training sessions."
        data.append({
            "instruction": f"'{question_16}'.",
            "input": f"\"Is there training available for using the {service} at ISQ?\"",
            "output": f"\"{answer_16}\"",
        })

        question_17 = f"Can I find user guides or manuals for {service} online?"
        answer_17 = f"Yes, user guides and manuals for {service} are available online. You can also contact {responsible} at {phone} or {email} for assistance."
        data.append({
            "instruction": f"'{question_17}'.",
            "input": f"\"Where can I find user guides for the {service} service at ISQ?\"",
            "output": f"\"{answer_17}\"",
        })

        question_18 = f"Are there any upcoming events related to {service} at ISQ?"
        answer_18 = f"To stay updated on upcoming events related to {service}, contact {responsible} at {phone} or {email}. You can also check our website for event announcements."
        data.append({
            "instruction": f"'{question_18}'.",
            "input": f"\"Are there any upcoming events related to the {service} service at ISQ?\"",
            "output": f"\"{answer_18}\"",
        })

        question_19 = f"Can I request a customized solution or service from ISQ?"
        answer_19 = f"Yes, we offer customized solutions. To discuss your specific needs, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_19}'.",
            "input": f"\"Can I request a customized solution or service from ISQ?\"",
            "output": f"\"{answer_19}\"",
        })

        question_20 = f"How often is the {service} service updated or improved?"
        answer_20 = f"We continuously strive to update and improve the {service} service. For the latest updates, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_20}'.",
            "input": f"\"How often is the {service} service at ISQ updated or improved?\"",
            "output": f"\"{answer_20}\"",
        })

        question_21 = f"Can you provide testimonials or references for the {service} service at ISQ?"
        answer_21 = f"Yes, we can provide testimonials or references for the {service} service upon request. Contact {responsible} at {phone} or {email} for more information."
        data.append({
            "instruction": f"'{question_21}'.",
            "input": f"\"Can you provide testimonials or references for the {service} service at ISQ?\"",
            "output": f"\"{answer_21}\"",
        })

        question_23 = f"Can I request a demo or trial for the {service} service before making a commitment?"
        answer_23 = f"Yes, we offer demos or trials for the {service} service. To request one, please contact {responsible} at {phone} or {email} for arrangements."
        data.append({
            "instruction": f"'{question_23}'.",
            "input": f"\"Can I request a demo or trial for the {service} service at ISQ?\"",
            "output": f"\"{answer_23}\"",
        })

        question_24 = f"Are there any specific requirements or qualifications needed to access the {service} service at ISQ?"
        answer_24 = f"To access the {service} service, specific requirements or qualifications may apply. Contact {responsible} at {phone} or {email} for detailed information."
        data.append({
            "instruction": f"'{question_24}'.",
            "input": f"\"Are there any specific requirements or qualifications needed to access the {service} service at ISQ?\"",
            "output": f"\"{answer_24}\"",
        })

        question_25 = f"What is the typical response time for inquiries or support requests related to {service} at ISQ?"
        answer_25 = f"Our typical response time for inquiries or support requests related to {service} is [insert response time here]. For urgent matters, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_25}'.",
            "input": f"\"What is the typical response time for inquiries or support requests related to the {service} service at ISQ?\"",
            "output": f"\"{answer_25}\"",
        })

        question_26 = f"Can I find case studies or success stories related to the {service} service at ISQ?"
        answer_26 = f"Yes, we have case studies and success stories related to the {service} service. Contact {responsible} at {phone} or {email} to request this information."
        data.append({
            "instruction": f"'{question_26}'.",
            "input": f"\"Can I find case studies or success stories related to the {service} service at ISQ?\"",
            "output": f"\"{answer_26}\"",
        })

        question_28 = f"Can I collaborate with ISQ on research or projects related to the {service} service?"
        answer_28 = f"Yes, we welcome collaborations on research or projects related to the {service} service. Contact {responsible} at {phone} or {email} to discuss potential opportunities."
        data.append({
            "instruction": f"'{question_28}'.",
            "input": f"\"Can I collaborate with ISQ on research or projects related to the {service} service?\"",
            "output": f"\"{answer_28}\"",
        })

        question_29 = f"Is there a customer portal or online account for managing {service} service-related activities?"
        answer_29 = f"Yes, we have a customer portal for managing {service} service-related activities. You can log in or register on our website. For further assistance, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_29}'.",
            "input": f"\"Is there a customer portal or online account for managing {service} service-related activities at ISQ?\"",
            "output": f"\"{answer_29}\"",
        })

        question_30 = f"Are there any ongoing promotions or discounts for the {service} service at ISQ?"
        answer_30 = f"For information on ongoing promotions or discounts for the {service} service, contact {responsible} at {phone} or {email}. You can also check our website or subscribe to our newsletter for updates."
        data.append({
            "instruction": f"'{question_30}'.",
            "input": f"\"Are there any ongoing promotions or discounts for the {service} service at ISQ?\"",
            "output": f"\"{answer_30}\"",
        })

        question_31 = f"Can you provide documentation or technical specifications for {service} at ISQ?"
        answer_31 = f"Yes, we have documentation and technical specifications for {service}. Contact {responsible} at {phone} or {email} to request this information."
        data.append({
            "instruction": f"'{question_31}'.",
            "input": f"\"Can you provide documentation or technical specifications for the {service} service at ISQ?\"",
            "output": f"\"{answer_31}\"",
        })

        question_32 = f"Is the {service} service compliant with industry standards or regulations?"
        answer_32 = f"Yes, the {service} service is compliant with industry standards and regulations. For specific details, contact {responsible} at {phone} or {email}."
        data.append({
            "instruction": f"'{question_32}'.",
            "input": f"\"Is the {service} service at ISQ compliant with industry standards or regulations?\"",
            "output": f"\"{answer_32}\"",
        })

        question_33 = f"Can I download brochures or promotional materials for the {service} service?"
        answer_33 = f"Yes, you can download brochures and promotional materials for the {service} service from our website. Additionally, contact {responsible} at {phone} or {email} for further assistance."
        data.append({
            "instruction": f"'{question_33}'.",
            "input": f"\"Can I download brochures or promotional materials for the {service} service at ISQ?\"",
            "output": f"\"{answer_33}\"",
        })

        question_34 = f"Is there a subscription or membership option for accessing premium features of the {service} service?"
        answer_34 = f"Yes, we offer subscription or membership options for accessing premium features of the {service} service. Contact {responsible} at {phone} or {email} to learn more about available plans."
        data.append({
            "instruction": f"'{question_34}'.",
            "input": f"\"Is there a subscription or membership option for accessing premium features of the {service} service at ISQ?\"",
            "output": f"\"{answer_34}\"",
        })

        question_35 = f"Can I get a quote or estimate for the {service} service at ISQ?"
        answer_35 = f"Yes, you can request a quote or estimate for the {service} service by contacting {responsible} at {phone} or {email}. Provide details about your specific requirements for an accurate quote."
        data.append({
            "instruction": f"'{question_35}'.",
            "input": f"\"Can I get a quote or estimate for the {service} service at ISQ?\"",
            "output": f"\"{answer_35}\"",
        })

        question_36 = f"Is there a self-service option or online resources for troubleshooting issues with {service}?"
        answer_36 = f"Yes, we provide self-service options and online resources for troubleshooting issues with {service}. Visit our support page on the website or contact {responsible} at {phone} or {email} for assistance."
        data.append({
            "instruction": f"'{question_36}'.",
            "input": f"\"Is there a self-service option or online resources for troubleshooting issues with the {service} service at ISQ?\"",
            "output": f"\"{answer_36}\"",
        })

        question_37 = f"Can I subscribe to notifications or updates for the {service} service at ISQ?"
        answer_37 = f"Yes, you can subscribe to notifications or updates for the {service} service by providing your email address on our website. Contact {responsible} at {phone} or {email} for additional information."
        data.append({
            "instruction": f"'{question_37}'.",
            "input": f"\"Can I subscribe to notifications or updates for the {service} service at ISQ?\"",
            "output": f"\"{answer_37}\"",
        })

        question_38 = f"Is there a customer satisfaction survey for feedback on the {service} service?"
        answer_38 = f"Yes, we conduct customer satisfaction surveys for feedback on the {service} service. Your opinions are valuable to us. Contact {responsible} at {phone} or {email} for more details on participating."
        data.append({
            "instruction": f"'{question_38}'.",
            "input": f"\"Is there a customer satisfaction survey for feedback on the {service} service at ISQ?\"",
            "output": f"\"{answer_38}\"",
        })

        question_39 = f"Are there any partnerships or collaborations between ISQ and other organizations for the {service} service?"
        answer_39 = f"Yes, we have partnerships and collaborations with other organizations for the {service} service. Contact {responsible} at {phone} or {email} for information on current collaborations."
        data.append({
            "instruction": f"'{question_39}'.",
            "input": f"\"Are there any partnerships or collaborations between ISQ and other organizations for the {service} service?\"",
            "output": f"\"{answer_39}\"",
        })

        question_40 = f"Can I access historical data or records related to the {service} service at ISQ?"
        answer_40 = f"Yes, you can request access to historical data or records related to the {service} service. Contact {responsible} at {phone} or {email} for assistance with your specific request."
        data.append({
            "instruction": f"'{question_40}'.",
            "input": f"\"Can I access historical data or records related to the {service} service at ISQ?\"",
            "output": f"\"{answer_40}\"",
        })

        question_41 = f"ISQ have any department responsible for the {service}?"
        answer_41 = f"Yes, the responsible for department responsible for the {service} is {depart}."
        data.append({
            "instruction": f"'{question_41}'.",
            "input": f"\"Can I access the department responsible for the {service} service at ISQ?\"",
            "output": f"\"{answer_41}\"",
        })

        question_41_alternative = f"Does ISQ have a department dedicated to {service}?"
        answer_41_alternative = f"Yes, the department responsible for {service} at ISQ is {depart}."
        data.append({
            "instruction": f"'{question_41_alternative}'.",
            "input": f"\"Is there a department at ISQ that handles the {service} service?\"",
            "output": f"\"{answer_41_alternative}\"",
        })

        question_42 = f"Is there a dedicated department at ISQ for {service} management?"
        answer_42 = f"Yes, ISQ has a specific department responsible for managing the {service} service, and it is {depart}."
        data.append({
            "instruction": f"'{question_42}'.",
            "input": f"\"Can I find a department at ISQ specifically handling {service} management?\"",
            "output": f"\"{answer_42}\"",
        })

        question_43 = f"Could you inform me if there is any department within ISQ overseeing the {service} service?"
        answer_43 = f"Yes, there is a department at ISQ overseeing the {service} service, and it is {depart}."
        data.append({
            "instruction": f"'{question_43}'.",
            "input": f"\"I'm wondering if there's a department at ISQ that oversees the {service} service. Can you provide information on that?\"",
            "output": f"\"{answer_43}\"",
        })

        question_44 = f"Is there a specific department within ISQ that deals with {service}?"
        answer_44 = f"Yes, there is a department at ISQ specifically handling {service}, and it is {depart}."
        data.append({
            "instruction": f"'{question_44}'.",
            "input": f"\"I'd like to know if there's a department at ISQ dealing with {service}.\"",
            "output": f"\"{answer_44}\"",
        })

        question_45 = f"Could you inform me about the department at ISQ responsible for {service}?"
        answer_45 = f"Certainly, the department responsible for {service} at ISQ is {depart}."
        data.append({
            "instruction": f"'{question_45}'.",
            "input": f"\"I'm looking for information on the department at ISQ responsible for {service}.\"",
            "output": f"\"{answer_45}\"",
        })
    responsavel_counts = df['Responsável de serviço'].value_counts()
    email_counts = df['Email de contacto'].value_counts()
    phone_counts = df['Telefone'].value_counts()

    print(responsavel_counts)
    print("\n\n\n")
    print(email_counts)
    print("\n\n\n")
    print(phone_counts)
    print("\n\n\n")
    responsible_arr = list(set(responsible_arr))

    # print(service_arr)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    
    # print(responsible_arr)
    return service_arr
    

if __name__ == "__main__":
    excel_path = "servicosISQ_tudo.xlsx"
    output_file = "datasets/ignore.json"
    service_arr = generate_questions(excel_path, output_file)
    # print(service_arr)
    # show_services(excel_path, output_file, service_arr)


# Frequência que cada responsável de serviço aparece:
# João Pombo                    95
# Maria Manuel Farinha          49
# João Brás Luis                24
# Sandra Isabel Fernandes       18
# Paula Gorjão                  18
# Ana Cristina Gouveia          12
# Rui Mendes                    10
# Frazão Guerreiro               9
# Alexandre Levy                 8
# Rui A. Louro                   7
# Carlos Martins                 6
# Pedro Pinto                    6
# Luís Ferreira                  5
# Maria João Franco              5
# Januário da Torre              5
# Tânia Farinha                  5
# Elsa Maria Cantiga             4
# André Ramalho                  3
# Cristina Leão                  3
# Rogério Magalhães              3
# Liliana P. Silva               3
# Sara Leonardo                  3
# Luis Conde Santos              3
# Vasco Mendes Pires             3
# João Paulo Figueiredo          2
# Elsa Cantiga                   2
# José Azevedo                   2
# Jorge Silva                    1
# Hugo Carrasqueira              1
# Carlos Almas Ramos             1
# Liliana Silva                  1
# Manuel Amorim                  1
# Filipe Lopes                   1
# Nuno Marques                   1
# Paulo Chaves                   1
# Carla Caetano                  1
# Luís Guedes / Tânia Santos     1
# Luís Guedes                    1
# Vasco Pires                    1
# José Barros                    1
# Natália Ribeiro                1
#  Vasco Pires                   1
# Marta Freitas                  1
# César Ribeiro                  1
# Segurança e Ambiente           1
# Nelson Fernandes               1