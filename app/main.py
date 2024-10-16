import os
from langchain_community.llms import Together
from langchain_community.embeddings import VoyageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory 

import streamlit as st
from uiconfig import UiConfig
from flask import Flask, session 
# Set API keys
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
VOYAGE_API_KEY = st.secrets["VOYAGE_API_KEY"]

# Initialize the LLM
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.5,
    max_tokens=500,
    top_k=20
)

# Initialize the embedding model
embeddings = VoyageEmbeddings()

# Load FAISS index which we have already created and saved
@st.cache_resource(show_spinner=False)
def load_faiss_index(save_directory="app/index_store_BRSR"):
    faiss_index = FAISS.load_local(save_directory, embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Function to generate hypothetical documents based on a query
def generate_hypothetical_documents(query, llm):
    prompt_template = f"Based on the query: '{query}', generate a hypothetical document or scenario that could answer the query."
    llm_result = llm.generate([prompt_template], max_tokens=500)
    hypothetical_document = llm_result.generations[0][0].text
    return hypothetical_document

# Function to convert hypothetical documents to embeddings
def generate_hypothetical_embeddings(hypothetical_document, embeddings):
    hypothetical_embeddings = embeddings.embed_documents([hypothetical_document])
    return hypothetical_embeddings

# Function to retrieve real documents using hypothetical embeddings
def retrieve_real_documents(hypothetical_embeddings, faiss_index):
    real_docs = faiss_index.similarity_search_by_vector(hypothetical_embeddings[0], k=5)  # Top 5 results
    return real_docs

# Function to run the HyDE process and then pass it to the QA chain
def run_hyde_query(query, llm, faiss_index, embeddings, qa_chain):
    # Step 1: Generate Hypothetical Document
    hypothetical_doc = generate_hypothetical_documents(query, llm)

    # Step 2: Convert Hypothetical Document to Embeddings
    hypothetical_embeddings = generate_hypothetical_embeddings(hypothetical_doc, embeddings)

    # Step 3: Retrieve Real Documents using FAISS
    retrieved_docs = retrieve_real_documents(hypothetical_embeddings, faiss_index)

    # Combine retrieved documents into a context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    #  Combining context and question into a single input
    # input_data = f"Context:\n{context}\n\nQuestion: {query}"
    input_data = f"Context:\n{context}\n\nMemory:\n{memory.load_memory_variables({})['chat_history']}\n\nQuestion: {query}"

    # Step 4: Generate the final response using the QA chain
    # final_answer = qa_chain.run({"context": context, "question": query})
    # final_answer = qa_chain.run({"input": input_data})
    final_answer = qa_chain.run({"input": input_data})

    return final_answer

# Add memory for conversation tracking
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(llm=llm, 
                                                        memory_key= "chat_history", 
                                                        return_messages= True)
memory = st.session_state.memory
# Add memory for conversation tracking
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_memory_length = 3)

# Main function for chatbot workflow
def main():
    UiConfig.setup()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant", avatar="🤖"):
        st.write("Hi, How can I help you!!")

    # Load FAISS index
    faiss_index = load_faiss_index()

    # Build the prompt for question and answer
    # qa_system_prompt = """You are an ESG intelligence expert assistant, you play a pivotal role as an expert and consultant specializing in ESG reporting and filing, particularly in the context of the BRSR Framework. Your primary function is to provide comprehensive solutions for users seeking ESG insights and facilitate competitor analysis.

    # Your task is to answer user queries using the provided context, which may include text and tables. In case the response involves a table, present it in a well-organized Markdown format. Similarly, for comparative analyses, ensure that the results are delivered in a clean, structured Markdown table.

    # When tasked with trend analysis, dive deep into the data, offering clear and detailed insights supported by evidence. Your responses should be meticulous, concise, and highly informative, reflecting your expertise as a one-stop solution for ESG inquiries.

    # It's imperative to avoid any form of hallucination in your responses, maintaining clarity and precision. Do not begin answers with generic references; instead, respond as an expert, ensuring your insights are detailed and directly beneficial to the user's ESG-related queries.

    # In instances where a question falls outside the provided context or if you lack the necessary information, simply acknowledge that you don't have the answer. Your overall goal is to serve as a reliable and knowledgeable resource, delivering expert guidance for users seeking ESG intelligence and analysis.
    # {context}
    # Question: {question}
    # """

    qa_system_prompt = """You are an ESG intelligence expert assistant, you play a pivotal role as an expert and consultant specializing in ESG reporting and filing, particularly in the context of the BRSR Framework. Your primary function is to provide comprehensive solutions for users seeking ESG insights and facilitate competitor analysis.

    Your task is to answer user queries using the provided context, which may include text and tables. In case the response involves a table, present it in a well-organized Markdown format. Similarly, for comparative analyses, ensure that the results are delivered in a clean, structured Markdown table.

    When tasked with trend analysis, dive deep into the data, offering clear and detailed insights supported by evidence. Your responses should be meticulous, concise, and highly informative, reflecting your expertise as a one-stop solution for ESG inquiries.

    It's imperative to avoid any form of hallucination in your responses, maintaining clarity and precision. Do not begin answers with generic references; instead, respond as an expert, ensuring your insights are detailed and directly beneficial to the user's ESG-related queries.

    In instances where a question falls outside the provided context or if you lack the necessary information, simply acknowledge that you don't have the answer. Your overall goal is to serve as a reliable and knowledgeable resource, delivering expert guidance for users seeking ESG intelligence and analysis.
    {input}
    """

    # prompt = PromptTemplate(template=qa_system_prompt, input_variables=["context", "question"])
    prompt = PromptTemplate(template=qa_system_prompt, input_variables=["input"])

    # # Add memory for conversation tracking
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_memory_length = 3)

    # Create an LLMChain for answering questions using the retrieved context and memory
    qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask the question"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👥"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                final_answer = run_hyde_query(query, llm, faiss_index, embeddings, qa_chain)
                message_placeholder.markdown(final_answer)

                # Log memory content after every query
        memory_data = memory.load_memory_variables({})
        print("Memory contents:", memory_data)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()
