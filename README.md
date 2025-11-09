# LLM Data Assistant
![Status](https://img.shields.io/badge/status-production--ready-success)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Days](https://img.shields.io/badge/days-42%2F42-green)
![Commits](https://img.shields.io/badge/commits-150+-orange)
![Tests](https://img.shields.io/badge/tests-50+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

AI-powered data analysis assistant using Large Language Models (Groq) and Retrieval-Augmented Generation.

🌐 **[Try the Live Demo]()** | 📚 **[Week 5 Summary](WEEK5_SUMMARY.md)**

## 🎯 What's New in v2.1.0

- ✅ **Modern Web Interface** - Beautiful Streamlit app with dark theme
- ✅ **Natural Language Chat** - Ask questions about your data in plain English
- ✅ **Latest AI Models** - Groq Llama 3.1-8B-Instant integration
- ✅ **Smart Data Analysis** - Automatic insights and report generation
- ✅ **Export Functionality** - Download conversations and reports
- ✅ **Mobile Responsive** - Works perfectly on all devices


### Branches
- `main` - Production-ready code (tagged releases only)
- `dev` - Active development (daily commits)


## Quick Start
```bash
# Clone
git clone https://github.com/ahmedyasir779/llm-data-assistant.git
cd llm-data-assistant

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
# Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" > .env

# Create sample data
python create_sample_data.py

# Run app
streamlit run app.py
```

## Get Groq API Key (FREE!)

1. Visit: https://console.groq.com
2. Sign up (no credit card needed)
3. Get API key
4. Add to `.env` file

## Tech Stack

- **LLM Provider:** Groq (FREE)
- **Model:** qwen/qwen3-32b
- **Language:** Python 3.9+
- **Framework:** Custom (learning-focused)
- **Coming:** ChromaDB, Docker

## UI Improvements:
- Beautiful dark chat bubbles (user: dark blue, assistant: dark gray)
- Modern CSS with gradients and animations
- Responsive design with proper spacing
- Quick action buttons with icons
- Enhanced sidebar with dataset info
- Export conversation functionality

## Sample Data:
- products.csv: 20 realistic e-commerce products
- reviews.csv: 50 customer reviews with ratings
- sales.csv: 1,500 transactions across 90 days
- All datasets include realistic, clean data

## Related Projects

- [Month 1: Data & Text Pipeline](https://github.com/ahmedyasir779/data-text-pipeline) - NLP & Data Processing (Live: https://data-text-pipeline.onrender.com)


## Author

**Ahmed Yasir**
- GitHub: [@ahmedyasir779](https://github.com/ahmedyasir779)
- LinkedIn: www.linkedin.com/in/ahmed-yasir-907561206
- Building in public | Shipping every week

## License

MIT License - Free to use and learn from

---

**Status:** 🟢 Active Development
#
