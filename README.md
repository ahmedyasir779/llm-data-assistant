# LLM Data Assistant

AI-powered data analysis assistant using Large Language Models (Groq) and Retrieval-Augmented Generation.

## Project Goal

Build an intelligent assistant that can:
- Analyze datasets using natural language
- Answer questions about your data
- Generate insights and summaries
- Provide data cleaning recommendations
- Create visualizations based on requests

### Branches
- `main` - Production-ready code (tagged releases only)
- `dev` - Active development (daily commits)

# Tag version
git tag -a v2.X.0 -m "Week X complete"

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
cp .env.example .env
# Add your Groq API key to .env

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
**Last Updated:** Day 29 - LLM integration working
