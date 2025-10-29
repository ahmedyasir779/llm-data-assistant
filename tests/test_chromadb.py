"""
LLM Data Assistant - Enhanced Version
Day 35: Advanced Features + ChromaDB Preview
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.enhanced_llm_client import EnhancedLLMClient
from src.smart_visualizer import SmartVisualizer
from src.chromadb_preview import DataVectorStore

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
    st.session_state.vector_store = DataVectorStore()
if "use_rag" not in st.session_state:
    st.session_state.use_rag = False


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


def get_comprehensive_data_info(df: pd.DataFrame, file_name: str) -> str:
    """
    Get comprehensive data information including actual data samples
    This is what the AI needs to answer questions!
    """
    info_parts = []
    
    # Basic info
    info_parts.append(f"📄 FILE: {file_name}")
    info_parts.append(f"📊 SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    info_parts.append(f"📋 COLUMNS: {', '.join(df.columns.tolist())}")
    
    # Data types
    info_parts.append(f"\n🔤 DATA TYPES:")
    for col, dtype in df.dtypes.items():
        info_parts.append(f"  - {col}: {dtype}")
    
    # Sample data (CRITICAL - This is what was missing!)
    info_parts.append(f"\n📝 SAMPLE DATA (first 10 rows):")
    info_parts.append(df.head(10).to_string())
    
    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        info_parts.append(f"\n📈 STATISTICS:")
        info_parts.append(df[numeric_cols].describe().to_string())
    
    # Value counts for categorical columns (if not too many unique values)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() <= 20:  # Only show if reasonable number of unique values
            info_parts.append(f"\n🏷️ VALUE COUNTS for '{col}':")
            info_parts.append(df[col].value_counts().head(10).to_string())
    
    # Missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        info_parts.append(f"\n⚠️ MISSING VALUES:")
        for col, count in null_counts[null_counts > 0].items():
            info_parts.append(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    return "\n".join(info_parts)


def create_data_analysis_prompt(all_datasets_info: str, question: str) -> list:
    """Create a comprehensive prompt with all data"""
    messages = [
        {
            "role": "system",
            "content": """You are an expert data analyst assistant. You have access to the complete datasets below.

IMPORTANT RULES:
1. Answer questions based on the ACTUAL DATA provided
2. When asked about specific values (max, min, average, etc.), calculate from the data shown
3. Be specific with numbers - don't give generic answers
4. If asked to compare, analyze the actual rows provided
5. Suggest visualizations when appropriate
6. Be conversational and helpful

FORMAT YOUR ANSWERS:
- Use bullet points for lists
- Include specific numbers and values
- Mention row/column names when relevant
- Keep answers clear and concise"""
        },
        {
            "role": "user",
            "content": f"""Here is the complete dataset information:

{all_datasets_info}

QUESTION: {question}

Please analyze the data above and provide a specific, detailed answer based on the actual values shown."""
        }
    ]
    
    return messages


# Sidebar
with st.sidebar:
    st.title("🤖 LLM Data Assistant")
    st.caption("v2.2.0 - Day 35 Fixed")
    
    st.divider()
    
    # File upload
    st.subheader("📁 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload datasets to analyze"
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
                    st.metric("Columns", len(df.columns))
                with col2:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
                    st.metric("Nulls", df.isnull().sum().sum())
                
                if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                    del st.session_state.datasets[name]
                    st.rerun()
    
    st.divider()
    
    # RAG toggle
    st.subheader("⚙️ Advanced")
    use_rag = st.toggle(
        "🔍 Enable RAG",
        value=st.session_state.use_rag,
        help="Use ChromaDB for semantic search"
    )
    
    if use_rag != st.session_state.use_rag:
        st.session_state.use_rag = use_rag
        if use_rag:
            for name, df in st.session_state.datasets.items():
                st.session_state.vector_store.add_dataframe_context(df, name)
            st.success("✅ RAG enabled")
    
    st.divider()
    
    if st.button("🔄 Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.datasets = {}
        st.session_state.vector_store.reset()
        st.rerun()


# Main content
st.title("💬 Chat with Your Data")

if not st.session_state.datasets:
    st.info("""
    👋 **Welcome to LLM Data Assistant!**
    
    **To get started:**
    1. Upload a CSV or Excel file in the sidebar
    2. Ask questions about your data
    3. Get instant insights and visualizations
    
    **Example questions:**
    - "What is the highest price?"
    - "Show me products with rating above 4.5"
    - "What's the average sales per product?"
    - "Create a chart showing trends"
    """)
    
    # Quick start with sample data
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Products Sample", use_container_width=True):
            sample_df = pd.DataFrame({
                'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
                'price': [1200, 25, 75, 300, 150],
                'sales': [50, 200, 150, 80, 120],
                'rating': [4.5, 4.2, 4.7, 4.6, 4.3]
            })
            st.session_state.datasets['products_sample.csv'] = sample_df
            st.rerun()
    
    with col2:
        if st.button("💰 Sales Sample", use_container_width=True):
            sample_df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10),
                'revenue': [1000, 1200, 950, 1100, 1300, 1150, 1400, 1250, 1350, 1500],
                'customers': [45, 52, 41, 49, 58, 51, 62, 55, 60, 67]
            })
            st.session_state.datasets['sales_sample.csv'] = sample_df
            st.rerun()
    
    with col3:
        if st.button("⭐ Reviews Sample", use_container_width=True):
            sample_df = pd.DataFrame({
                'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
                'rating': [4.5, 4.2, 4.7, 4.6, 4.3],
                'reviews': [125, 340, 210, 180, 95]
            })
            st.session_state.datasets['reviews_sample.csv'] = sample_df
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show visualizations if present
        if "chart" in message and message["chart"]:
            st.plotly_chart(message["chart"], use_container_width=True)

# Chat input (only if data is loaded)
if st.session_state.datasets:
    # Quick action buttons
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Show Columns", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "What are the column names in my datasets?"
            })
            st.rerun()
    
    with col2:
        if st.button("📊 Summary Stats", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Give me summary statistics"
            })
            st.rerun()
    
    with col3:
        if st.button("📈 Create Chart", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Create an appropriate visualization"
            })
            st.rerun()
    
    with col4:
        if st.button("🔍 Data Quality", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Check data quality and report any issues"
            })
            st.rerun()
    
    st.divider()
    
    # Chat form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What is the highest price?",
            key="user_input"
        )
        submit_button = st.form_submit_button("Send 🚀", use_container_width=True)
    
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        with st.spinner("🤔 Analyzing data..."):
            # Build comprehensive data context with ACTUAL DATA
            all_datasets_info = []
            
            for name, df in st.session_state.datasets.items():
                dataset_info = get_comprehensive_data_info(df, name)
                all_datasets_info.append(dataset_info)
            
            combined_info = "\n\n" + "="*80 + "\n\n".join(all_datasets_info)
            
            # Use RAG if enabled
            if st.session_state.use_rag:
                rag_context = st.session_state.vector_store.get_relevant_context(user_input)
                combined_info = f"RAG CONTEXT:\n{rag_context}\n\n{combined_info}"
            
            # Create proper prompt with data
            messages = create_data_analysis_prompt(combined_info, user_input)
            
            # Get response from LLM
            response = st.session_state.llm_client.chat(messages)
            
            # Check if visualization requested
            chart = None
            if any(word in user_input.lower() for word in ['chart', 'plot', 'graph', 'visualize', 'show']):
                # Create visualization
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

# Footer
st.divider()
st.caption("Built with ❤️ by Ahmed Yasir")
st.caption("GitHub: [@ahmedyasir779](https://github.com/ahmedyasir779) | [LinkedIn](https://www.linkedin.com/in/ahmed-yasir-907561206)")