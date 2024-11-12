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
pc = Pinecone(api_key=os.getenv('PINE_CONE_KEY'))
index = pc.Index("vector-embeddings")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key = st.secrets['OPENAI_API_KEY'])
system_instruction = "You are a helpful assistant."

logo = "./G.png"

# App title and layout
st.set_page_config(
    page_title="Google Chat Bot",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar components
st.sidebar.markdown(
    "<h1 style='font-size:35px; color:white;'>🤖 Pixel AI Bot</h1>", 
    unsafe_allow_html=True
)

# Initialize model and memory length
if 'conversational_memory_length' not in st.session_state:
    st.session_state.conversational_memory_length = 6  # Default value

model = st.sidebar.selectbox(
    'Choose a Model',
    ['llama3-70b-8192', 'llama-3.1-70b-versatile']
)

# Sidebar slider for conversational memory length
conversational_memory_length = st.sidebar.slider('Conversational Memory Length:', 1, 10, value=st.session_state.conversational_memory_length)

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "What's on your mind today?"}]
    st.session_state.memory = ConversationBufferWindowMemory(k=st.session_state.conversational_memory_length)  # Reset memory

# Button to clear chat history
if st.sidebar.button('Clear Chat History'):
    clear_chat_history()

# Expander in sidebar
with st.sidebar.expander("Some Quick Prompts for Reference"):
    st.write("List all the Google Smart Phones")
    st.write("Compare Pixel 6 and Pixel 6a in tabular format")
    st.write("How to connect Ear Buds")
    st.write("What are the features of Thermostat")

# Initialize messages if not already in session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "What's on your mind today?"}]  # Set initial message

# Initialize memory if not already in session state
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferWindowMemory(k=st.session_state.conversational_memory_length)

# Main app function
def main():
    memory = st.session_state['memory']

    # Initialize ChatGroq with the selected model
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model_name=model
    )

    # Display chat history if there are any messages
    if st.session_state['messages']:
        st.markdown("<h3>Chatbot Interaction for Google Products:</h3>", unsafe_allow_html=True)
        for message in st.session_state['messages']:
            role = message['role']
            content = message['content']
            if role == 'assistant':
                st.write(f"**🤖 Pixel Bot:** {content}")
            else:
                st.markdown(f'<span style="color:orange"><strong>🧑 You:</strong> {content}</span>', unsafe_allow_html=True)

    # Chat input at the bottom of the page
    user_question = st.chat_input("Ask a question...")

    # If the user submits a question
    if user_question:
        # Append user question to chat history
        st.session_state['messages'].append({"role": "user", "content": user_question})

        # Display the user's question immediately
        st.markdown(f'<span style="color:orange"><strong>🧑 You:</strong> {user_question}</span>', unsafe_allow_html=True)

        # Use spinner while processing the response
        with st.spinner("Extracting..."):
            # Retrieve documents based on the query
            doc_search = retrieve_query(user_question)

            # Generate answer from retrieved documents
            generated_answer = answer_from_doc_search(doc_search, user_question, llm)

            # Append assistant's response to chat history
            st.session_state['messages'].append({"role": "assistant", "content": generated_answer or "No response generated."})

        # Rerun the app to update the display
        st.rerun()


def retrieve_query(query, k = 10 ):
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
