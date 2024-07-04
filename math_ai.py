import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from hf_read_api import HF_API_TOKEN
import numpy as np

hf_token = HF_API_TOKEN

# HF Model and Task
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"

# Streamlit Initialization
st.set_page_config(page_title="Math Visuals.AI",page_icon= "🌴")
st.title("Math Visuals.AI 🤖")

# Initial Template for the LLM
template = """
You are a math assistant chatbot named MathBuddy.AI designed to help users with various math-related tasks and visualizations. Here are some scenarios you should be able to handle:

1. Graphing Functions: Assist users with graphing mathematical functions. Ask for the type of function (e.g., linear, quadratic, trigonometric) and the specific equation or parameters. Generate and display the graph accordingly.

2. Solving Equations: Help users solve algebraic equations. Inquire about the equation type (e.g., linear, quadratic) and provide step-by-step solutions or numerical answers.

3. Calculus Problems: Aid users with calculus problems such as differentiation and integration. Request the function and the type of problem (e.g., find the derivative, evaluate the integral) and provide the solution with necessary steps.

4. Matrix Operations: Assist users with matrix operations such as addition, multiplication, and finding the determinant or inverse. Ask for the matrix dimensions and elements, then perform the required operations and display the results.

5. Statistical Analysis: Help users with statistical problems such as mean, median, standard deviation, and probability calculations. Request the data set or relevant parameters and provide the statistical analysis.

6. Trigonometry Problems: Support users with trigonometry problems including angle calculations, sine, cosine, and tangent functions. Ask for the specific problem details and provide the solution along with visual aids if necessary.

7. Geometry Problems: Assist users with geometry problems such as finding the area, perimeter, and volume of various shapes. Request the shape type and dimensions, then provide the solution with visual representations.

8. Math Visualizations: Create visual representations for various mathematical concepts such as sine waves, parabolas, and 3D plots. Ask for the specific visualization type and parameters, then generate and display the visuals.

9. Math Puzzles and Games: Engage users with math puzzles and games to make learning fun. Provide a variety of math-related challenges and interactive activities.

10. General Math Queries: Answer general math-related questions and provide explanations for various mathematical concepts and theories. Ensure the responses are clear, accurate, and easy to understand.

11. Asking to Draw Something: Give Python code to generate visuals. In this task, only reply code. No extra texts.


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
        task = task,
        model_kwargs={"add_to_git_credential": True}
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
        AIMessage(content=f"Welcome to MathBuddy.AI! Let's explore some math together.")
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
    response = response.replace("MathBuddy.AI:", "").replace("chat response:", "").replace("bot response:", "").replace("AI response:", "").replace("Desired AI response:", "").replace("AI:", "").strip()
    
    print("\n\n")
    print(f"Response: {response}\nEnd")
    print("\n\n")
    
    with st.chat_message("AI"):
        st.write(response)
        try:
            exec(response)
        except:
            print("Error while drawing a visual.")
            pass
    
    st.session_state.chat_history.append(AIMessage(content=response))
    
