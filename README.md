This repository aims to develop a chatbot.

Currently, the language used for communication between the user and the bot must be in English. 

It is used Langchain (https://python.langchain.com/docs/get_started/installation)

Langchain is a tool that allows you to create and manage language-based AI applications. Here is a step-by-step breakdown of what Langchain does:

• Environment Setup: Install the Langchain CLI and set up the necessary environment variables.

• Create a Langchain Project: Use the Langchain CLI to create a new project or add Langchain to an existing project.

• Configure LangSmith (Optional): If you have access to LangSmith, a tracing and monitoring tool for Langchain applications, you can configure it by setting the necessary environment variables.

• Start LangServe: Run the Langchain CLI command to start the FastAPI app, which serves as the backend for your Langchain application.

• Access Templates: You can view all the available templates and access the documentation and playground for each template.

• Access Templates from Code: You can interact with the Langchain templates programmatically by importing the necessary modules and creating a runnable instance.

• Usage: Depending on the specific template you're using, you can perform various tasks such as question-answering, language generation, and more.

• Retrieve Information: Langchain uses retrieval techniques to gather relevant information from the web or other sources based on the given question or context.

• Process and Generate Responses: Langchain leverages large language models (LLMs) to process the retrieved information and generate accurate and informative responses.

• Chain of Thought: Langchain allows for the chaining of multiple steps or actions to handle complex tasks and provide comprehensive answers.

• Debug and Monitor: If you have access to LangSmith, you can use it to trace, monitor, and debug your Langchain applications.

It is necessary to input the .xlsx file, which contains the relevant data. 


Notas:


O Langchain é uma plataforma que permite a criação e execução de cadeias de ferramentas de processamento de linguagem natural (NLP) de forma automatizada. Ele combina várias ferramentas e modelos de NLP para realizar tarefas específicas, como pesquisa na web, tradução, geração de respostas, análise de sentimentos, entre outras. O Langchain permite que os usuários criem fluxos de trabalho personalizados, chamados de "cadeias", que podem ser executados em sequência para obter resultados desejados. Essas cadeias podem ser usadas para automatizar tarefas complexas de processamento de linguagem natural, economizando tempo e esforço.

O Langchain utiliza técnicas de aprendizado de máquina, mais especificamente modelos de linguagem de grande escala (LLMs, na sigla em inglês). Esses modelos são treinados em grandes quantidades de texto para aprender padrões e estruturas da linguagem. Eles são capazes de gerar texto coerente e realizar tarefas de processamento de linguagem natural, como tradução, resumo de texto, geração de respostas, entre outras. O Langchain aproveita esses modelos de linguagem para fornecer funcionalidades avançadas de processamento de linguagem natural em suas cadeias de ferramentas.

Uma LLM (Large Language Model) é um modelo de linguagem de grande escala. É um tipo de modelo de aprendizado de máquina treinado em uma enorme quantidade de texto para aprender a estrutura e os padrões da linguagem. Esses modelos são capazes de gerar texto coerente e realizar tarefas de processamento de linguagem natural, como tradução, resumo de texto, geração de respostas, entre outras. Eles são treinados em grandes conjuntos de dados, como a internet, para capturar a diversidade e complexidade da linguagem humana. As LLMs são usadas em várias aplicações de processamento de linguagem natural devido à sua capacidade de compreender e gerar texto de forma semelhante aos humanos.

Uma LLM (Large Language Model) é uma rede neural, mais especificamente um tipo de modelo de linguagem baseado em redes neurais. Esses modelos são construídos com arquiteturas de redes neurais profundas, como redes neurais recorrentes (RNNs) ou transformers. Eles são treinados em grandes conjuntos de dados textuais usando técnicas de aprendizado supervisionado para aprender a prever a próxima palavra em uma sequência de texto. Essa capacidade de previsão permite que a LLM gere texto coerente e realize tarefas de processamento de linguagem natural. Portanto, uma LLM é uma rede neural e não uma random forest.s
The ideia is to create a data.txt file "conversation" examples between the user and the chatbot. For example:
"User: Hello
Chabot: Hello! How can i help you?
User: What is the capital of Portugal?
Chatbot: The capital of Portugal is Lisbon"

