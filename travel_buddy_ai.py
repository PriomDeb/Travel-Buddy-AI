import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from hf_read_api import HF_API_TOKEN
    hf_token = HF_API_TOKEN
except:
    os.environ["hf_token"] == st.secrets["HF_TOKEN"]
    hf_token = os.environ["hf_token"]


# HF Model and Task
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"

# Streamlit Initialization
st.set_page_config(page_title="TravelBuddy.AI",page_icon= "🌴")
st.title("🌴 Travel Buddy AI 🤖")

# Initial Template for the LLM
template = """
You are a  travel assistant chatbot your name is Travel Buddy.AI designed to help users plan their trips and provide travel-related information. Here are some scenarios you should be able to handle:

1. Booking Flights: Assist users with booking flights to their desired destinations. Ask for departure city, destination city, travel dates, and any specific preferences (e.g., direct flights, airline preferences). Check available airlines and book the tickets accordingly.

2. Booking Hotels: Help users find and book accommodations. Inquire about city or region, check-in/check-out dates, number of guests, and accommodation preferences (e.g., budget, amenities). 

3. Booking Rental Cars: Facilitate the booking of rental cars for travel convenience. Gather details such as pickup/drop-off locations, dates, car preferences (e.g., size, type), and any additional requirements.

4. Destination Information: Provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit.

5. Travel Tips: Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.

6. Weather Updates: Give current weather updates for specific destinations or regions. Include temperature forecasts, precipitation chances, and any weather advisories.

7. Local Attractions: Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.

8. Customer Service: Address customer service inquiries and provide assistance with travel-related issues. Handle queries about bookings, cancellations, refunds, and general support.

Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

Chat history:
{chat_history}

User question:
{user_question}
"""

prompt = ChatPromptTemplate.from_template(template)

def llm_response(user_query, chat_history):
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token = HF_API_TOKEN,
        repo_id = repo_id,
        task = task
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke(
        {
            "chat_history": chat_history,
            "user_question": user_query,
        }
    )
    
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=f"✨ Ready to embark on your next adventure? Let’s make your travel dreams come true! From planning the perfect itinerary to uncovering hidden gems, I’m here to ensure your journey is seamless and unforgettable. Where shall we explore today? ✈️🌟")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Ask me anything...")

# Get User Input
if user_query and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    response = llm_response(user_query, st.session_state.chat_history)
    
    # Removing any prefixes from the response
    response = response.replace("AI reponse:", "").replace("chat response:", "").replace("bot response:", "").strip()
    
    with st.chat_message("AI"):
        st.write(response)
    
    st.session_state.chat_history.append(AIMessage(content=response))
    
