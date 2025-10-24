from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant that answers questions based on provided reviews.
You have access to the following reviews: {reviews}
Here are some questions to answer: {questions}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = prompt | model

while True:
    print("/n/n------------------------------------------------")
    user_input = input("Enter reviews and questions (or 'exit' to quit): ")
    print("/n/n------------------------------------------------")
    if user_input.lower() == 'exit':
        break

    reviews = retriever.invoke(user_input)
    results = chain.invoke({"reviews":reviews,"questions": [user_input]})
    print(results)