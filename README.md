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

# Test
python src/llm_client.py
```

## Get Groq API Key (FREE!)

1. Visit: https://console.groq.com
2. Sign up (no credit card needed)
3. Get API key
4. Add to `.env` file

## Tech Stack

- **LLM Provider:** Groq (FREE)
- **Model:** Llama 3 (8B parameters)
- **Language:** Python 3.9+
- **Framework:** Custom (learning-focused)
- **Coming:** ChromaDB, Streamlit, Docker

## Learning Resources

This project follows structured learning:
- Week 5: LLM fundamentals
- Week 6: Prompt engineering
- Week 7: RAG implementation
- Week 8: Production deployment

## Related Projects

- [Month 1: Data & Text Pipeline](https://github.com/ahmedyasir779/data-text-pipeline) - NLP & Data Processing (Live: https://data-text-pipeline.onrender.com)

## Progress
```
Week 5: [███░░░░░] Day 1/7 (14%)
Month 2: [█░░░░░░░] Week 1/4 (25%)
Roadmap: [████░░░░] Month 2/4 (50%)
```

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
