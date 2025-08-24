import streamlit as st
import asyncio
import os
import sys

# Add the parent directory to the path to import from 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retriever import knowledge_base

def main():
    """
    Streamlit UI for testing the KnowledgeBase retriever.
    """
    st.set_page_config(page_title="Retriever Test UI", layout="wide")
    st.title("🔬 Knowledge Base Retriever Test")
    st.write("Enter a query to search the vectorized documents.")

    # Get user input
    query = st.text_input("Enter your query:", "What are the latest advancements in starship technology?")

    if st.button("Retrieve"):
        if not query:
            st.warning("Please enter a query.")
            return

        with st.spinner("Retrieving information from the knowledge base..."):
            try:
                # Run the async retrieve function
                results = asyncio.run(knowledge_base.retrieve(query))
                
                st.success("Retrieval complete!")
                
                if results:
                    st.subheader("Retrieved Context:")
                    # Display the entire formatted string directly
                    st.info(results) 
                else:
                    st.info("No relevant documents found for your query.")

            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")

if __name__ == "__main__":
    # This allows running the async main function within Streamlit's sync environment
    main()
