import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_chat import DataChat


# Page configuration
st.set_page_config(
    page_title="LLM Data Assistant",
    page_icon="Dee",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with dark chat bubbles
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Dark chat bubbles */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        color: white;
        border-left: 4px solid #3498db;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        color: white;
        border-left: 4px solid #27ae60;
        margin-right: 2rem;
    }
    
    .user-message strong {
        color: #74b9ff;
    }
    
    .assistant-message strong {
        color: #00cec9;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Input styling */
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa, #ffffff);
    }
    
    /* Success/info styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #cce7ff, #b3d9ff);
        border-left: 4px solid #007bff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_chat' not in st.session_state:
        st.session_state.data_chat = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display modern app header"""
    st.markdown('<h1 class="main-header">🤖 LLM Data Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your data using natural language powered by Groq Llama 3.1</p>', unsafe_allow_html=True)


def display_welcome():
    """Display welcome screen when no data is loaded"""
    
    # Welcome message with icon
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgb(102, 126, 234), rgb(190, 168, 212)); border-radius: 12px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 1rem;"> Welcome to LLM Data Assistant</h2>
        <p style="color: #666; font-size: 1.1rem;">Upload a data file to start chatting with your data!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 210px;">
            <h3 style="color: #3498db; margin-bottom: -0.7rem;"> Upload Data</h3>
            <ul style="color: #666;">
                <li>CSV files</li>
                <li>Excel files (.xlsx, .xls)</li>
                <li>JSON files</li>
                <li>Automatic analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 210px;">
            <h3 style="color: #27ae60; margin-bottom: -0.7rem;"> Chat Naturally</h3>
            <ul style="color: #666;">
                <li>"How many rows?"</li>
                <li>"What's the average price?"</li>
                <li>"Tell me about ratings"</li>
                <li>"Show top products"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 210px;">
            <h3 style="color: #e74c3c; margin-bottom: -0.7rem;"> Get Insights</h3>
            <ul style="color: #666;">
                <li>AI-powered analysis</li>
                <li>Statistical summaries</li>
                <li>Pattern detection</li>
                <li>Export results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Example datasets section
    st.markdown("---")
    st.markdown("### 📁 Try Example Datasets")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if Path('data/products.csv').exists():
            if st.button("🛍️ Load Products Dataset (20 items)", key="load_products"):
                load_example_dataset('data/products.csv', 'products.csv')
    
    with example_col2:
        if Path('data/sales.csv').exists():
            if st.button("💰 Load Sales Dataset (1800+ records)", key="load_sales"):
                load_example_dataset('data/sales.csv', 'sales.csv')


def load_example_dataset(filepath: str, filename: str):
    """Load example dataset"""
    try:
        with st.spinner(f"Loading {filename}..."):
            st.session_state.data_chat = DataChat(filepath)
            st.session_state.uploaded_file_name = filename
            st.session_state.chat_history = []
            st.session_state.message_count = 0
        st.success(f"✓ Loaded {filename}")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error loading example: {e}")


def display_data_overview(data_chat: DataChat):
    """Display data overview with modern styling"""
    st.markdown("### 📊 Dataset Overview")
    
    metadata = data_chat.loader.metadata
    
    # Metrics in a nice grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Rows", f"{metadata['rows']:,}")
    
    with col2:
        st.metric("📋 Columns", metadata['columns'])
    
    with col3:
        missing_count = sum(info['count'] for info in metadata['missing_values'].values())
        st.metric("❌ Missing", missing_count)
    
    with col4:
        numeric_cols = len(metadata['statistics'])
        st.metric("🔢 Numeric", numeric_cols)
    
    # Expandable sections with better styling
    with st.expander("📋 Column Details", expanded=False):
        col_df = pd.DataFrame({
            'Column': metadata['column_names'],
            'Type': [metadata['dtypes'][col] for col in metadata['column_names']],
            'Missing': [metadata['missing_values'].get(col, {}).get('count', 0) for col in metadata['column_names']]
        })
        st.dataframe(col_df, width="stretch", hide_index=True)
    
    with st.expander("📈 Quick Statistics", expanded=False):
        if metadata['statistics']:
            stats_data = []
            for col, stats in metadata['statistics'].items():
                stats_data.append({
                    'Column': col,
                    'Mean': round(stats['mean'], 2),
                    'Median': round(stats['median'], 2),
                    'Min': round(stats['min'], 2),
                    'Max': round(stats['max'], 2)
                })
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width="stretch", hide_index=True)
        else:
            st.info("No numeric columns available")
    
    with st.expander("👀 Sample Data", expanded=False):
        sample_df = pd.DataFrame(metadata['sample_data'])
        st.dataframe(sample_df, width="stretch", hide_index=True)


def display_chat_interface(data_chat: DataChat):
    """Display modern chat interface"""
    st.markdown("### 💬 Chat with Your Data")
    
    # Chat container with scrolling
    chat_container = st.container()
    
    with chat_container:
        # Display chat history with dark bubbles
        for i, message in enumerate(st.session_state.chat_history):
            role = message['role']
            content = message['content']
            
            if role == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Truncate very long responses
                display_content = content
                if len(content) > 1000:
                    display_content = content[:1000] + "..."
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br><br>
                    {display_content}
                </div>
                """, unsafe_allow_html=True)
    
    # Divider
    st.markdown("---")
    
    # Quick action buttons
    st.markdown("**⚡ Quick Actions:**")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("📋 Show Columns", key="btn_columns"):
            add_quick_response("What columns are in the dataset?", get_columns_info(data_chat))
    
    with action_col2:
        if st.button("📊 Basic Stats", key="btn_stats"):
            add_quick_response("Show me basic statistics", get_basic_stats(data_chat))
    
    with action_col3:
        if st.button("🔍 Data Summary", key="btn_summary"):
            add_quick_response("Summarize this dataset", get_data_summary(data_chat))
    
    with action_col4:
        if st.button("🔄 Clear Chat", key="btn_clear"):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.rerun()
    
    # Chat input - FIXED to prevent double sends
    st.markdown("**💭 Ask a question:**")
    
    # Use a form to prevent accidental double submissions
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "",
            placeholder="e.g., How many rows? What's the average price? Tell me about the data...",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Send 🚀", width="stretch")
        
        if submitted and user_question.strip() and not st.session_state.processing:
            process_user_question(user_question.strip(), data_chat)


def add_quick_response(question: str, response: str):
    """Add a quick response to chat history"""
    st.session_state.chat_history.append({
        'role': 'user',
        'content': question
    })
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response
    })
    st.session_state.message_count += 2
    st.rerun()


def get_columns_info(data_chat: DataChat) -> str:
    """Get formatted columns information"""
    metadata = data_chat.loader.metadata
    columns_info = "📋 **Dataset Columns:**\n\n"
    
    for i, col in enumerate(metadata['column_names'], 1):
        dtype = metadata['dtypes'][col]
        missing = metadata['missing_values'].get(col, {}).get('count', 0)
        columns_info += f"{i}. **{col}** ({dtype})"
        if missing > 0:
            columns_info += f" - {missing} missing"
        columns_info += "\n"
    
    return columns_info


def get_basic_stats(data_chat: DataChat) -> str:
    """Get formatted basic statistics"""
    metadata = data_chat.loader.metadata
    missing_total = sum(info['count'] for info in metadata['missing_values'].values())
    
    stats_info = f"""📊 **Dataset Statistics:**

- **Total Rows:** {metadata['rows']:,}
- **Total Columns:** {metadata['columns']}
- **Missing Values:** {missing_total}
- **Numeric Columns:** {len(metadata['statistics'])}

**File:** {metadata['name']}
"""
    
    if metadata['statistics']:
        stats_info += "\n**Numeric Column Ranges:**\n"
        for col, stats in list(metadata['statistics'].items())[:3]:  # Show first 3
            stats_info += f"• **{col}:** {stats['min']:.1f} - {stats['max']:.1f} (avg: {stats['mean']:.1f})\n"
    
    return stats_info


def get_data_summary(data_chat: DataChat) -> str:
    """Get AI-generated data summary"""
    try:
        return data_chat.generator.summarize_dataset(data_chat.loader.metadata)
    except Exception as e:
        return f"📊 **Dataset Summary:**\n\nThis dataset contains {data_chat.loader.metadata['rows']} rows and {data_chat.loader.metadata['columns']} columns. The data includes both numeric and categorical information that can be analyzed for patterns and insights.\n\n*Error generating detailed summary: {str(e)[:100]}...*"


def process_user_question(question: str, data_chat: DataChat):
    """Process user question with improved error handling"""
    # Set processing flag
    st.session_state.processing = True
    
    # Add user message
    st.session_state.chat_history.append({
        'role': 'user',
        'content': question
    })
    st.session_state.message_count += 1
    
    # Show thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("🤔 **Thinking...**")
    
    try:
        # Get response from AI with timeout
        response = data_chat.chat.send_message(question)
        
        # Clean up response (remove weird tokens if any)
        cleaned_response = clean_response(response)
        
        # Add assistant response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': cleaned_response
        })
        st.session_state.message_count += 1
        
    except Exception as e:
        # Fallback response
        error_response = f"Sorry, I encountered an issue processing your question. Let me try to help with basic information about your dataset instead.\n\n{get_basic_stats(data_chat)}"
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': error_response
        })
        st.session_state.message_count += 1
    
    finally:
        # Clear processing flag and thinking indicator
        st.session_state.processing = False
        thinking_placeholder.empty()
        st.rerun()


def clean_response(response: str) -> str:
    """Clean response from potential gibberish tokens"""
    # List of common gibberish tokens to filter out
    gibberish_tokens = [
        'Basel', 'Toastr', 'PSI', 'RODUCTION', 'Injected', 'Britain', 
        'exposition', 'contaminants', 'roscope', 'BuilderFactory',
        'externalActionCode', 'visitInsn', 'dateTime', 'slider', '_both'
    ]
    
    # If response contains too many gibberish tokens, return a fallback
    gibberish_count = sum(1 for token in gibberish_tokens if token in response)
    
    if gibberish_count > 3:  # If more than 3 gibberish tokens
        return "I apologize, but I'm having trouble generating a clear response. Could you please rephrase your question or try asking something more specific about your dataset?"
    
    return response


def sidebar_file_upload():
    """Enhanced sidebar file upload"""
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; text-align: center;">📁 Upload Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Upload CSV, Excel, or JSON file (max 200MB)"
    )
    
    if uploaded_file is not None:
        # Check if it's a new file
        if st.session_state.uploaded_file_name != uploaded_file.name:
            try:
                # Create temp directory
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                temp_file_path = temp_dir / uploaded_file.name
                
                # Save file
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load with progress
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                status_text.text("📂 Saving file...")
                progress_bar.progress(25)
                
                status_text.text("🔍 Analyzing data...")
                progress_bar.progress(50)
                
                # Initialize DataChat
                st.session_state.data_chat = DataChat(str(temp_file_path))
                progress_bar.progress(75)
                
                # Reset chat
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.message_count = 0
                progress_bar.progress(100)
                
                status_text.text("✅ Ready!")
                time.sleep(1)
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.sidebar.success(f"✓ Loaded {uploaded_file.name}")
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)[:50]}...")


def sidebar_info():
    """Enhanced sidebar info"""
    if st.session_state.data_chat is not None:
        st.sidebar.markdown("---")
        
        # Dataset info
        metadata = st.session_state.data_chat.loader.metadata
        st.sidebar.markdown(f"""
        <div style="background: rgb(14, 17, 23); padding: 1rem; border-radius: 8px;">
            <h4 style="margin-top: 0;"> Current Dataset</h4>
            <p><strong>File:</strong> {metadata['name']}</p>
            <p><strong>Size:</strong> {metadata['rows']:,} × {metadata['columns']}</p>
            <p><strong>Messages:</strong> {st.session_state.message_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Export option
        if st.session_state.chat_history:
            st.sidebar.markdown("### 💾 Export")
            if st.sidebar.button("📥 Download Chat"):
                export_conversation()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: rgb(14, 17, 23); padding: 1rem; border-radius: 8px; text-align: center;">
        <h4 style="margin-top: 0;">🤖 About</h4>
        <p><strong>LLM Data Assistant</strong></p>
        <p>Powered by Groq Llama 3.1</p>
        <p><em>Week 5, Day 34</em></p>
    </div>
    """, unsafe_allow_html=True)


def export_conversation():
    """Export conversation as JSON"""
    if st.session_state.chat_history:
        conversation_data = {
            'dataset': st.session_state.uploaded_file_name,
            'timestamp': datetime.now().isoformat(),
            'message_count': st.session_state.message_count,
            'messages': st.session_state.chat_history
        }
        
        json_str = json.dumps(conversation_data, indent=2)
        
        st.sidebar.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main application with error handling"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display header
        display_header()
        
        # Sidebar
        sidebar_file_upload()
        sidebar_info()
        
        # Main content
        if st.session_state.data_chat is None:
            # Welcome screen
            display_welcome()
        else:
            # Data loaded - show chat interface
            col1, col2 = st.columns([1, 2])
            
            with col1:
                display_data_overview(st.session_state.data_chat)
            
            with col2:
                display_chat_interface(st.session_state.data_chat)
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()