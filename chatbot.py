import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# Initialize Pinecone Client
pc = Pinecone(api_key = os.getenv('PINE_CONE_KEY'))
index = pc.Index("vector-embeddings")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key = st.secrets['OPENAI_API_KEY'])

system_instruction = "You are a helpful assistant."

logo = "./G.png"

# App title and layout
st.set_page_config(page_title="Google Chat Bot", page_icon=logo, layout="wide")

# Sidebar components
#st.sidebar.title("🤖Pixel AI Bot")
st.sidebar.markdown(
    "<h1 style='font-size:35px; color:white;'>🤖 Pixel AI Bot</h1>", 
    unsafe_allow_html=True
)

model = st.sidebar.selectbox(
    'Choose a Model',
    ['llama3-70b-8192', 'llama-3.1-70b-versatile']
)

conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=6)

# Function to reset chat history and memory
def clear_chat_history():
    st.session_state['messages'] = []
    st.session_state['memory'] = ConversationBufferWindowMemory(k=conversational_memory_length)
    st.session_state['query'] = ""  # Clear the query input box

# Button to clear chat history
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Google products prompts
st.sidebar.title("Quick Prompts")
google_prompts = [
    "List all the Google Smart Phones.",
    "Compare Pixel 6 and Pixel 6a in tabular format",
    "What are the features of Thermostat",
    "What is the Charger type of Phones",
    "How to connect Ear Buds"
]

# Define a callback to update the query input when a quick prompt is selected
def update_query():
    st.session_state.query = st.session_state.selected_prompt

# Add the selectbox with an on_change callback to update the query
st.sidebar.selectbox(
    "Choose a Quick Prompt here", google_prompts, key="selected_prompt", on_change=update_query)

# Initialize memory if not already in session state
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(k=conversational_memory_length)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'input_submitted' not in st.session_state:
    st.session_state['input_submitted'] = False

# Main app function
def main():

    memory = st.session_state['memory']

    # Initialize ChatGroq with the selected model
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name=model
    )

    # Conditionally set the query default value to an empty string after submission
    query_default = "" if st.session_state['input_submitted'] else st.session_state.get("query", "")
    
    # Full-width text area with placeholder
    user_question = st.text_area(
        "", placeholder="Ask a question...", 
        value=query_default, height=80, key="query"
    )

    # Add "Enter" button for submitting the prompt
    enter_button = st.button("Submit")

    # If the enter button is pressed, execute the RAG process
    if enter_button and user_question:
        # Append user question to chat history
        st.session_state['messages'].append({"role": "user", "content": user_question})

        # Retrieve documents based on the query
        doc_search = retrieve_query(user_question)

        # Generate answer from retrieved documents
        generated_answer = answer_from_doc_search(doc_search, user_question, llm)

        # Append assistant's response to chat history
        st.session_state['messages'].append({"role": "assistant", "content": generated_answer or "No response generated."})

        # Set flag to reset the query input field
        st.session_state['input_submitted'] = True

    else:
        # Reset input submission flag for future interactions
        st.session_state['input_submitted'] = False

    # Display chat history if there are any messages
    if st.session_state['messages']:
        st.markdown("<h3>Chatbot Interaction:</h3>", unsafe_allow_html=True)
        for message in st.session_state['messages']:
            role = message['role']
            content = message['content']
            if role == 'assistant':
                st.write(f"**🤖 Pixel Bot:** {content}")
            else:
                st.markdown(f'<span style="color:orange"><strong>🧑 You:</strong> {content}</span>', unsafe_allow_html=True)
                #st.write(f"**🧑 You:** {content}")

def retrieve_query(query, k=3):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
    matching_results = [result['metadata']['text'] for result in results['matches']]
    return matching_results

def generate_text_groq(prompt, llm, max_tokens=400):
    response = llm.invoke(prompt)
    return response.content

def answer_from_doc_search(doc_search, query, llm):
    combined_docs = "\n\n".join(doc for doc in doc_search)
    prompt = f"{system_instruction}\n\nAnswer the question concisely and accurately based on the following documents.\n\nQuestion: {query}\n\nDocuments:\n{combined_docs}"
    answer = generate_text_groq(prompt, llm)
    return answer

if __name__ == "__main__":
    main()
