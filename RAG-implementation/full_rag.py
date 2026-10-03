import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

# get api keys

load_dotenv()
qdrant_api = os.getenv("QDRANT_API_KEY")
groq_api = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

#connect to qdrant
client = QdrantClient(
    url = qdrant_url,
    api_key = qdrant_api
)


# create new collection
collection_name = "knowleage_base"
embadding_size = 384

if client.collection_exists(collection_name):
    print(f"Deleting collection {collection_name}")
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config = VectorParams(
        size=embadding_size,
        distance=Distance.COSINE
    ),
)
print(f"Created collection: {collection_name}")
print(f"Vector size: {embadding_size}")
print("Distance: COSINE")

# LOAD OUR KNOWLEDGE
with open("requrment.txt" , "r") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents")

#creat embedding 

model = SentenceTransformer("all-MiniLM-L6-v2")
embaddings = model.encode(documents) #here embedding is array

# create qdrant points
points = []
for i,embadding in enumerate(embaddings):
    point = PointStruct(
        id = i+1,
        vector = embadding.tolist(),
        payload = {
            "text" : documents[i]
        }
    )
    points.append(point)

# upload this point on qdrant 

client.upsert(
    collection_name = collection_name,
    points = points
)

# searching
def searching(query , top_k = 3):
    query_vector = model.encode(query).tolist()
    result = client.query_points(
        collection_name=collection_name,
        query = query_vector,
        limit = top_k,
        with_payload = True
    ).points
    return result 

#connect to groq
groq_client = Groq(api_key=groq_api)

def ask_llm(question,context):
    prompt = f"""
    Answer the question using only the information provided below.

    Context:
    {context}

    Question:
    {question}

    If the answer is not present in the context, say:
    "I don't know based on the provided information.
    """ 
    groq_model = "openai/gpt-oss-120b"

    response = groq_client.chat.completions.create(
        model = groq_model,
        messages = [
            {
                "role" : "user",
                "content" : prompt
            }
        ]
    )

    answer = response.choices[0].message.content
    return answer

question = "How many vacation days do i get?"
results = searching(question,top_k=3)

context = "\n".join(
    result.payload["text"]
    for result in results
)

ans = ask_llm(question,context)
print(ans)