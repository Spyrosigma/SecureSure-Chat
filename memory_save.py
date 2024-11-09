from pinecone import Pinecone
import requests
import os , uuid
from dotenv import load_dotenv
load_dotenv()

JINA_API_KEY = os.environ.get('JINA_API_KEY')

def memory_upload(memory_text, Topic):
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index = pc.Index("mymemory")
    try:
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {JINA_API_KEY}'
        }
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "dimensions": 1024,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [memory_text,Topic]
        }
        response = requests.post(url, headers=headers, json=data)
        memory_vector = response.json()['data'][0]['embedding']
        print("Memory Vector: \n ",memory_vector)

        index.upsert(
            vectors=[{
                'id': str(uuid.uuid1()) , # Unique Identifier ID
                'values':memory_vector,
                'metadata':{'Topic': Topic, 'text':memory_text}
            }],
            namespace='October'
        )
        return 'Done'
    except Exception as e:
        return f'Error: {e}'
