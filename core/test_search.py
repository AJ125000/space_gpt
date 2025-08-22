import streamlit as st
import sys
import os

# Add the current directory to the path to import tools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tools import web_search_tool
    tool_imported = True
except ImportError as e:
    tool_imported = False
    import_error = str(e)

def main():
    st.title("🔍 Web Search Tool - Serper.dev")
    st.markdown("---")
    
    # Check if tool was imported successfully
    if not tool_imported:
        st.error(f"❌ Failed to import web search tool: {import_error}")
        st.stop()
    
    st.success("✅ Web search tool imported successfully!")
    
    # Display tool information
    with st.expander("🔧 Tool Information", expanded=False):
        st.write(f"**Tool Name:** {web_search_tool.name}")
        st.write(f"**Description:** {web_search_tool.description}")
    
    # Main search interface
    st.markdown("### Enter your search query:")
    query = st.text_input(
        "Search Query", 
        placeholder="e.g., SpaceX latest launch news",
        help="Enter any search query to find current information"
    )
    
    # Search button
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    
    if search_button and query:
        with st.spinner(f"Searching for: '{query}'..."):
            try:
                # Call the web search tool
                results = web_search_tool.run(query)
                
                st.success(f"✅ Search completed for: **{query}**")
                st.markdown("---")
                
                # Display raw results in an expandable section
                with st.expander("📄 Raw Search Results", expanded=True):
                    st.text_area(
                        "Complete Response from Serper.dev:",
                        value=results,
                        height=400,
                        help="This shows the exact response from the Serper.dev API"
                    )
                
                # Try to parse and display formatted results if possible
                st.markdown("### 📊 Results Summary:")
                
                # Count approximate number of results by counting common indicators
                result_count = results.count('title') if 'title' in results else 0
                st.info(f"📈 Estimated number of results: {result_count}")
                
                # Display character and word count
                word_count = len(results.split())
                char_count = len(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("Tool Used", "Serper.dev")
                
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
                
    elif search_button and not query:
        st.warning("⚠️ Please enter a search query before clicking Search")
    
    # Example queries section
    st.markdown("---")
    st.markdown("### 💡 Example Queries:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 SpaceX News"):
            st.session_state.example_query = "SpaceX latest launch news"
            st.experimental_rerun()
    
    with col2:
        if st.button("🌌 NASA Updates"):
            st.session_state.example_query = "NASA latest space missions 2024"
            st.experimental_rerun()
            
    with col3:
        if st.button("🛰️ Space Technology"):
            st.session_state.example_query = "latest space technology breakthroughs"
            st.experimental_rerun()
    
    # Handle example queries
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        del st.session_state.example_query
        
        with st.spinner(f"Searching for: '{query}'..."):
            try:
                results = web_search_tool.run(query)
                
                st.success(f"✅ Search completed for: **{query}**")
                st.markdown("---")
                
                # Display results
                with st.expander("📄 Raw Search Results", expanded=True):
                    st.text_area(
                        "Complete Response from Serper.dev:",
                        value=results,
                        height=400
                    )
                
                # Results summary
                st.markdown("### 📊 Results Summary:")
                result_count = results.count('title') if 'title' in results else 0
                st.info(f"📈 Estimated number of results: {result_count}")
                
                word_count = len(results.split())
                char_count = len(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("Tool Used", "Serper.dev")
                    
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")

if __name__ == "__main__":
    main()