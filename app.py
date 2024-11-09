from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os, requests, uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "securesure"

JINA_API_KEY = os.environ.get("JINA_API_KEY")

llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama3-8b-8192",
    temperature=0.1,
)

@app.route("/")
def welcome():
    return "Welcome to the SecureSure Project Backend!"


@socketio.on("connect")
def handle_connect():
    user_id = str(uuid.uuid4())
    join_room(user_id)
    emit("set_user_id", {"user_id": user_id})
    print(f"Client connected with ID: {user_id}")


@socketio.on("join")
def on_join(data):
    user_id = data["user_id"]
    join_room(user_id)
    print(f"User {user_id} joined their room")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


user_conversations = {}


@socketio.on("user_message")
def handle_message(message):
    try:
        user_id = message["user_id"]
        user_query = message["data"]

        query = pc.inference.embed(
        "multilingual-e5-large", 
        inputs=[user_query], 
        parameters={"input_type": "query"}
        )

        index = pc.Index(index_name)
        results = index.query(
            namespace=index_name,
            vector=query[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True,
        )

        context = [
            results["matches"][0]["metadata"]["text"],
            results["matches"][1]["metadata"]["text"],
            results["matches"][2]["metadata"]["text"],
        ]
        context = " ".join(context)         
        print(f"User {user_id}: {user_query}")
        print("Results:", results)
        print("Pinecone Semantic Search :-- \n", context)

        system_prompt = f"""
        You are an expert Health Insurance Claims AI assistant with deep knowledge of insurance policies, medical billing, and claims processing. Your purpose is to help users understand and navigate their health insurance claims using the provided context: {context}

        CORE RESPONSIBILITIES:
        1. Claims Processing Support: Explain claim status, requirements, and procedures
        2. Policy Interpretation: Clarify coverage details, benefits, and limitations
        3. Billing Assistance: Help with understanding medical bills, EOBs, and payments
        4. Appeals Guidance: Explain the appeals process when claims are denied
        5. Respond user queries in short and to the point

        INTERACTION PROTOCOL:
        1. For each query, analyze:
        [QUERY] {user_query} [/QUERY]
        [CONTEXT] {context} [/CONTEXT]

        2. Response Generation:
    
        - If context matches query: Provide accurate, context-based information
        - If context partially matches: Use available information and clearly indicate any gaps
        - If context is missing/irrelevant: Request specific details or clarification
        - If query is unclear: Ask focused questions to understand user needs

        RESPONSE GUIDELINES:
        1. Accuracy:
        - Base responses strictly on provided context and verified insurance knowledge
        - Never speculate about specific claim details not present in context
        - Clearly distinguish between general insurance information and user-specific details

        2. Clarity:
        - Use simple, non-technical language
        - Break down complex insurance concepts
        - Format responses with bullet points or numbered lists for readability
        - Include relevant claim numbers, dates, and amounts when available

        3. Empathy:
        - Acknowledge user concerns about medical costs and coverage
        - Maintain a supportive, patient tone
        - Show understanding when dealing with claim denials or coverage issues

        BOUNDARIES:
        - Do not provide medical advice
        - Do not guarantee claim outcomes
        - Do not speculate about coverage not clearly stated in policy
        - Do not make promises about payment amounts or timeframes

        When unsure:
        1. Acknowledge limitations clearly
        2. Request specific information needed
        3. Direct users to appropriate resources (customer service, provider billing office, etc.)

        Keep responses concise and focused while ensuring all critical information is conveyed.
        """

        if user_id not in user_conversations:
            memory = ConversationBufferMemory(return_messages=True)
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
            user_conversations[user_id] = ConversationChain(
                llm=llm, memory=memory, prompt=prompt, verbose=False
            )

        conversation = user_conversations[user_id]
        response = conversation.predict(input=user_query)

        print("Groq: ", response)
        emit("bot_response", {"data": response}, room=user_id)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        emit("bot_response", {"data": error_message}, room=user_id)


if __name__ == "__main__":
    socketio.run(app, debug=True)
