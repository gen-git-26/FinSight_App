# FinSight 💹

**See Beyond The Numbers** - Smart Financial Agent with Real-Time Market Data

<div align="center">


An intelligent financial analysis agent that provides real-time market data, options analysis, fundamental research, and multi-company comparisons through a beautiful Streamlit interface.

[Features](#features) • [Installation](#installation) • [Configuration](#configuration) • [Usage](#usage) • [Architecture](#architecture)

</div>

---

##  Features

### Core Capabilities
- **Smart Query Understanding**: Natural language processing with LLM-powered ticker extraction
- **Multi-Ticker Comparison**: Compare financial metrics across multiple companies simultaneously
- **Real-Time Market Data**: Live stock prices, crypto quotes, and market information
- **News & Sentiment**: Latest financial news and market updates
- **Options Analysis**: Options chains, expiration dates, and derivatives data
- **Fundamental Analysis**: Balance sheets, income statements, cash flow analysis
- **Dynamic Routing**: Automatically selects the best data source for each query


### Supported Queries
```
✅ "Tesla stock price"
✅ "Compare the quarterly income statements of Amazon and Google"
✅ "AAPL options for 2024-12-20"
✅ "Meta Platforms news"
✅ "Bitcoin price"
✅ "What are the key financial metrics for Tesla"
✅ "NVDA vs AMD comparison"
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- Financial data API keys (Finnhub, Alpha Vantage)

### Step 1: Clone Repository
```bash
git clone <"https://github.com/gen-git-26/FinSight_App">
cd new_test
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install MCP Servers
```bash
# Install Yahoo Finance MCP
cd vendors/yahoo-finance-mcp
pip install -e .
cd ../..

# Install Financial Datasets MCP (if available)
cd vendors/financial-datasets-mcp
pip install -e .
cd ../..
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API (Required)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o  # or gpt-4, gpt-3.5-turbo

# Financial APIs (Optional - for fallback tools)
FINNHUB_API_KEY=your_finnhub_key
ALPHAVANTAGE_API_KEY=your_alphavantage_key
FINANCIAL_DATASETS_API_KEY=your_key  # If using financial-datasets MCP

# MCP Servers Configuration
MCP_SERVERS='[
  {
    "name": "yfinance",
    "command": "/path/to/.venv/bin/python",
    "args": ["-u", "/path/to/vendors/yahoo-finance-mcp/server.py"],
    "env": {"PYTHONUNBUFFERED": "1"},
    "cwd": "/path/to/vendors/yahoo-finance-mcp"
  },
  {
    "name": "financial-datasets",
    "command": "/path/to/.venv/bin/python",
    "args": ["-u", "/path/to/vendors/financial-datasets-mcp/server.py"],
    "env": {
      "PYTHONUNBUFFERED": "1",
      "FINANCIAL_DATASETS_API_KEY": "your_key"
    },
    "cwd": "/path/to/vendors/financial-datasets-mcp"
  }
]'
```

### 2. MCP Server Setup

**Important**: Replace `/path/to/` with your actual project path.

#### Option A: Automatic Path Detection
The system can auto-detect paths if MCP servers are in `vendors/`:
```bash
export MCP_SERVERS='[
  {
    "name": "yfinance",
    "command": "python",
    "args": ["-u", "vendors/yahoo-finance-mcp/server.py"]
  }
]'
```

#### Option B: Manual Configuration
For production, use absolute paths:
```bash
MCP_SERVERS='[{"name":"yfinance","command":"/home/user/project/.venv/bin/python","args":["-u","/home/user/project/vendors/yahoo-finance-mcp/server.py"]}]'
```

---

## 🎮 Usage

### Start the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Quick Queries**: Use the pre-built example buttons
   - "Ask me anything..."

2. **Custom Queries**: Type in the chat input at the bottom

3. **Live MCP Inspector**: Expand the "Live MCP" section to see:
   - Query routing details
   - Raw API responses
   - Parsed data structures

4. **View Sources**: Click "Sources" to see data attribution



## 🏗️ Architecture

### Project Structure
```
new_test/
├── app.py                      # Main Streamlit application
├── agent/
│   ├── __init__.py
│   └── agent.py               # Agent configuration & instructions
├── tools/
│   ├── __init__.py
│   ├── query_parser.py        # LLM-based query parsing
│   ├── mcp_router.py          # Dynamic MCP routing engine
│   ├── mcp_bridge.py          # Direct MCP access
│   ├── answer.py              # Response formatting
│   └── tools.py               # Legacy fallback tools
├── mcp_connection/
│   ├── __init__.py
│   ├── manager.py             # MCP server management
│   └── startup.py             # MCP lifecycle management
├── utils/
│   ├── config.py              # Configuration loader
│   └── logging.py             # Logging setup
├── vendors/                    # MCP servers
│   ├── yahoo-finance-mcp/
│   └── financial-datasets-mcp/
├── data/                       # Assets (logo, icons)
├── .env                        # Environment variables
└── requirements.txt            # Python dependencies








