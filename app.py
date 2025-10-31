import streamlit as st
import pandas as pd
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.enhanced_llm_client import EnhancedLLMClient
from src.smart_visualizer import SmartVisualizer
from src.chromadb_preview import DataVectorStore
from src.vector_store_advanced import AdvancedVectorStore
from src.rag_engine import RAGQueryEngine

# Page config
st.set_page_config(
    page_title="LLM Data Assistant v2.2",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);}
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "llm_client" not in st.session_state:
    try:
        st.session_state.llm_client = EnhancedLLMClient()
    except ValueError:
        st.error("⚠️ GROQ_API_KEY not found! Add it to .env file")
        st.stop()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = AdvancedVectorStore()  
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGQueryEngine(
        llm_client=st.session_state.llm_client,
        vector_store=st.session_state.vector_store
    )
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True  

def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_data_info(df: pd.DataFrame) -> dict:
    """Extract data information"""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_string(),
        "stats": df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "No numeric columns"
    }

# Sidebar
with st.sidebar:
    st.title("🤖 LLM Data Assistant")
    st.caption("v2.2.0 - Day 35")
    
    st.divider()
    
    # File upload
    st.subheader("📁 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.datasets:
                with st.spinner(f"Loading {uploaded_file.name}..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.datasets[uploaded_file.name] = df
                        
                        if st.session_state.use_rag:
                            st.session_state.vector_store.add_dataframe_context(df, uploaded_file.name)
                        
                        st.success(f"✅ Loaded {uploaded_file.name}")
    
    st.divider()
    
    # Dataset info
    if st.session_state.datasets:
        st.subheader("📊 Loaded Datasets")
        
        for name, df in st.session_state.datasets.items():
            with st.expander(f"📄 {name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                
                if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                    del st.session_state.datasets[name]
                    st.rerun()
    
    st.divider()
    
    # RAG toggle
    st.subheader("⚙️ Advanced RAG")
    use_rag = st.toggle(
        "🔍 Enable RAG Search",
        value=st.session_state.use_rag,
        help="Use ChromaDB vector store for semantic search"
    )

    
    if use_rag != st.session_state.use_rag:
        st.session_state.use_rag = use_rag
        if use_rag:
            for name, df in st.session_state.datasets.items():
                st.session_state.vector_store.add_dataframe_context(df, name)
            st.success("✅ RAG enabled - datasets indexed")
        else:
            st.info("ℹ️ RAG disabled - using direct data access")

    # Show vector store stats
    if st.session_state.use_rag:
        stats = st.session_state.vector_store.get_collection_stats()
        st.caption(f"📊 {stats['total_documents']} documents indexed")
    
    st.divider()
    
    if st.button("🔄 Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.datasets = {}
        st.session_state.vector_store.reset()
        st.rerun()

# Main content
st.title("💬 Chat with Your Data")

if not st.session_state.datasets:
    st.info("👋 Upload a CSV or Excel file to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Load Products Sample"):
            sample_df = pd.DataFrame({
                'product': ['Laptop', 'Mouse', 'Keyboard'],
                'price': [1200, 25, 75],
                'sales': [50, 200, 150]
            })
            st.session_state.datasets['products.csv'] = sample_df
            st.rerun()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "chart" in message and message["chart"]:
            st.plotly_chart(message["chart"], use_container_width=True)

# Chat input
if st.session_state.datasets:
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Columns"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are the columns?"
            })
            st.rerun()
    
    with col2:
        if st.button("📊 Stats"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Summary statistics"
            })
            st.rerun()
    
    with col3:
        if st.button("📈 Chart"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Create a chart"
            })
            st.rerun()
    
    with col4:
        if st.button("🔍 Quality"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Check data quality"
            })
            st.rerun()
    
    st.divider()
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is the average price?"
        )
        submit = st.form_submit_button("Send 🚀")
    
    if submit and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤔 Analyzing with RAG..." if st.session_state.use_rag else "🤔 Analyzing..."):
            if st.session_state.use_rag:
                # Use RAG engine
                response = st.session_state.rag_engine.query_with_rag(
                    user_input,
                    st.session_state.datasets,
                    n_context=3,
                    use_hybrid=True
                )
            else:
                # Use direct query
                response = st.session_state.rag_engine.query_without_rag(
                    user_input,
                    st.session_state.datasets
                )
            
            # Check if visualization requested
            chart = None
            if any(word in user_input.lower() for word in ['chart', 'plot', 'graph', 'visualize', 'show']):
                first_df = list(st.session_state.datasets.values())[0]
                visualizer = SmartVisualizer(first_df)
                chart = visualizer.auto_visualize()
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "chart": chart
            })
        
        st.rerun()

st.caption("LLM Data Assistant v2.3.0: Full RAG | Built with ❤️ by Ahmed Yasir")
st.caption("GitHub: [@ahmedyasir779](https://github.com/ahmedyasir779) | [LinkedIn](https://www.linkedin.com/in/ahmed-yasir-907561206)")