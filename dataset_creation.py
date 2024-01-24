import pandas as pd
import json
import numpy as np

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

        # question_19 = f"Can I request a customized solution or service from ISQ?"
        # answer_19 = f"Yes, we offer customized solutions. To discuss your specific needs, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_19}'.",
        #     "input": f"\"Can I request a customized solution or service from ISQ?\"",
        #     "output": f"\"{answer_19}\"",
        # })

        # question_20 = f"How often is the {service} service updated or improved?"
        # answer_20 = f"We continuously strive to update and improve the {service} service. For the latest updates, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_20}'.",
        #     "input": f"\"How often is the {service} service at ISQ updated or improved?\"",
        #     "output": f"\"{answer_20}\"",
        # })

        # question_21 = f"Can you provide testimonials or references for the {service} service at ISQ?"
        # answer_21 = f"Yes, we can provide testimonials or references for the {service} service upon request. Contact {responsible} at {phone} or {email} for more information."
        # data.append({
        #     "instruction": f"'{question_21}'.",
        #     "input": f"\"Can you provide testimonials or references for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_21}\"",
        # })

        # question_23 = f"Can I request a demo or trial for the {service} service before making a commitment?"
        # answer_23 = f"Yes, we offer demos or trials for the {service} service. To request one, please contact {responsible} at {phone} or {email} for arrangements."
        # data.append({
        #     "instruction": f"'{question_23}'.",
        #     "input": f"\"Can I request a demo or trial for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_23}\"",
        # })

        # question_24 = f"Are there any specific requirements or qualifications needed to access the {service} service at ISQ?"
        # answer_24 = f"To access the {service} service, specific requirements or qualifications may apply. Contact {responsible} at {phone} or {email} for detailed information."
        # data.append({
        #     "instruction": f"'{question_24}'.",
        #     "input": f"\"Are there any specific requirements or qualifications needed to access the {service} service at ISQ?\"",
        #     "output": f"\"{answer_24}\"",
        # })

        # question_25 = f"What is the typical response time for inquiries or support requests related to {service} at ISQ?"
        # answer_25 = f"Our typical response time for inquiries or support requests related to {service} is [insert response time here]. For urgent matters, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_25}'.",
        #     "input": f"\"What is the typical response time for inquiries or support requests related to the {service} service at ISQ?\"",
        #     "output": f"\"{answer_25}\"",
        # })

        # question_26 = f"Can I find case studies or success stories related to the {service} service at ISQ?"
        # answer_26 = f"Yes, we have case studies and success stories related to the {service} service. Contact {responsible} at {phone} or {email} to request this information."
        # data.append({
        #     "instruction": f"'{question_26}'.",
        #     "input": f"\"Can I find case studies or success stories related to the {service} service at ISQ?\"",
        #     "output": f"\"{answer_26}\"",
        # })

        # question_28 = f"Can I collaborate with ISQ on research or projects related to the {service} service?"
        # answer_28 = f"Yes, we welcome collaborations on research or projects related to the {service} service. Contact {responsible} at {phone} or {email} to discuss potential opportunities."
        # data.append({
        #     "instruction": f"'{question_28}'.",
        #     "input": f"\"Can I collaborate with ISQ on research or projects related to the {service} service?\"",
        #     "output": f"\"{answer_28}\"",
        # })

        # question_29 = f"Is there a customer portal or online account for managing {service} service-related activities?"
        # answer_29 = f"Yes, we have a customer portal for managing {service} service-related activities. You can log in or register on our website. For further assistance, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_29}'.",
        #     "input": f"\"Is there a customer portal or online account for managing {service} service-related activities at ISQ?\"",
        #     "output": f"\"{answer_29}\"",
        # })

        # question_30 = f"Are there any ongoing promotions or discounts for the {service} service at ISQ?"
        # answer_30 = f"For information on ongoing promotions or discounts for the {service} service, contact {responsible} at {phone} or {email}. You can also check our website or subscribe to our newsletter for updates."
        # data.append({
        #     "instruction": f"'{question_30}'.",
        #     "input": f"\"Are there any ongoing promotions or discounts for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_30}\"",
        # })

        # question_31 = f"Can you provide documentation or technical specifications for {service} at ISQ?"
        # answer_31 = f"Yes, we have documentation and technical specifications for {service}. Contact {responsible} at {phone} or {email} to request this information."
        # data.append({
        #     "instruction": f"'{question_31}'.",
        #     "input": f"\"Can you provide documentation or technical specifications for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_31}\"",
        # })

        # question_32 = f"Is the {service} service compliant with industry standards or regulations?"
        # answer_32 = f"Yes, the {service} service is compliant with industry standards and regulations. For specific details, contact {responsible} at {phone} or {email}."
        # data.append({
        #     "instruction": f"'{question_32}'.",
        #     "input": f"\"Is the {service} service at ISQ compliant with industry standards or regulations?\"",
        #     "output": f"\"{answer_32}\"",
        # })

        # question_33 = f"Can I download brochures or promotional materials for the {service} service?"
        # answer_33 = f"Yes, you can download brochures and promotional materials for the {service} service from our website. Additionally, contact {responsible} at {phone} or {email} for further assistance."
        # data.append({
        #     "instruction": f"'{question_33}'.",
        #     "input": f"\"Can I download brochures or promotional materials for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_33}\"",
        # })

        # question_34 = f"Is there a subscription or membership option for accessing premium features of the {service} service?"
        # answer_34 = f"Yes, we offer subscription or membership options for accessing premium features of the {service} service. Contact {responsible} at {phone} or {email} to learn more about available plans."
        # data.append({
        #     "instruction": f"'{question_34}'.",
        #     "input": f"\"Is there a subscription or membership option for accessing premium features of the {service} service at ISQ?\"",
        #     "output": f"\"{answer_34}\"",
        # })

        # question_35 = f"Can I get a quote or estimate for the {service} service at ISQ?"
        # answer_35 = f"Yes, you can request a quote or estimate for the {service} service by contacting {responsible} at {phone} or {email}. Provide details about your specific requirements for an accurate quote."
        # data.append({
        #     "instruction": f"'{question_35}'.",
        #     "input": f"\"Can I get a quote or estimate for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_35}\"",
        # })

        # question_36 = f"Is there a self-service option or online resources for troubleshooting issues with {service}?"
        # answer_36 = f"Yes, we provide self-service options and online resources for troubleshooting issues with {service}. Visit our support page on the website or contact {responsible} at {phone} or {email} for assistance."
        # data.append({
        #     "instruction": f"'{question_36}'.",
        #     "input": f"\"Is there a self-service option or online resources for troubleshooting issues with the {service} service at ISQ?\"",
        #     "output": f"\"{answer_36}\"",
        # })

        # question_37 = f"Can I subscribe to notifications or updates for the {service} service at ISQ?"
        # answer_37 = f"Yes, you can subscribe to notifications or updates for the {service} service by providing your email address on our website. Contact {responsible} at {phone} or {email} for additional information."
        # data.append({
        #     "instruction": f"'{question_37}'.",
        #     "input": f"\"Can I subscribe to notifications or updates for the {service} service at ISQ?\"",
        #     "output": f"\"{answer_37}\"",
        # })

        # question_38 = f"Is there a customer satisfaction survey for feedback on the {service} service?"
        # answer_38 = f"Yes, we conduct customer satisfaction surveys for feedback on the {service} service. Your opinions are valuable to us. Contact {responsible} at {phone} or {email} for more details on participating."
        # data.append({
        #     "instruction": f"'{question_38}'.",
        #     "input": f"\"Is there a customer satisfaction survey for feedback on the {service} service at ISQ?\"",
        #     "output": f"\"{answer_38}\"",
        # })

        # question_39 = f"Are there any partnerships or collaborations between ISQ and other organizations for the {service} service?"
        # answer_39 = f"Yes, we have partnerships and collaborations with other organizations for the {service} service. Contact {responsible} at {phone} or {email} for information on current collaborations."
        # data.append({
        #     "instruction": f"'{question_39}'.",
        #     "input": f"\"Are there any partnerships or collaborations between ISQ and other organizations for the {service} service?\"",
        #     "output": f"\"{answer_39}\"",
        # })

        # question_40 = f"Can I access historical data or records related to the {service} service at ISQ?"
        # answer_40 = f"Yes, you can request access to historical data or records related to the {service} service. Contact {responsible} at {phone} or {email} for assistance with your specific request."
        # data.append({
        #     "instruction": f"'{question_40}'.",
        #     "input": f"\"Can I access historical data or records related to the {service} service at ISQ?\"",
        #     "output": f"\"{answer_40}\"",
        # })

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

if __name__ == "__main__":
    excel_path = "servicosISQ_tudo.xlsx"
    output_file = "dataset_18qa_isq_desc.json"
    generate_questions(excel_path, output_file)
