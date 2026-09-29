import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key is not found")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


#rag 
# step-1 -> knowlage base

knowledge_base = {
    "age" : "nitin mehra's age is 22",
    "nitin" : "nitin mehra is a java developer",
}

# step-2  -> retrieval
def retrieve(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "nitin" in question:
        return knowledge_base["nitin"]
    else:
        return None


def run_agent(question :str):
    context = retrieve(question)
    if context is None:
        return "I don't know based on the available context."
    
    sys_message = {
        "role" : "system",
        "content" : f"""Answer in only one line and use only the context below. "
                    "If the answer is not present in the context, say you do not know. "
                    f"\n\nContext:\n{context}"""
    }
    message = {
        "role" : "user",
        "content" : question
    }
    messages = [sys_message , message]
    response = client.chat.completions.create(
        model= model,
        messages=messages
    )
    answer = response.choices[0].message.content
    return answer

question = "do you know nitin's age"
print(run_agent(question))