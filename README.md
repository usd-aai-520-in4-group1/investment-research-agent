# Investment Research Agent

A sophisticated AI-powered investment research system that combines LangGraph workflows, OpenAI's GPT models, and comprehensive financial data analysis to provide intelligent stock recommendations.

> **✨ NEW in v2.0**: Natural language queries ("What's the latest on Apple?") and dual-source data integration with automatic Yahoo Finance → Alpha Vantage fallback!

## Overview

This project is a complete investment research platform that leverages cutting-edge AI technology to analyze stocks comprehensively. It combines traditional financial analysis with modern AI capabilities to deliver actionable investment insights.

**Latest Updates (v2.0)**:
- 🗣️ **Natural Language Queries**: Extract stock symbols from plain English questions
- 🔄 **Multi-Source Data**: Automatic fallback between Yahoo Finance and Alpha Vantage
- 🛠️ **7 LangGraph Tools**: Extended tool ecosystem with Alpha Vantage integration

### Key Features

- **AI-Powered Analysis**: Uses OpenAI's GPT-4o-mini for intelligent financial analysis
- **Natural Language Queries**: 🆕 Ask questions in plain English - "What's the latest on Apple?" - and the agent automatically extracts the stock symbol
- **Multi-Source Data**: 🆕 Dual data source support with automatic Yahoo Finance → Alpha Vantage fallback for reliability
- **Agentic Workflows**: Three powerful workflow patterns for intelligent routing and optimization
  - **Prompt Chaining**: 5-step pipeline (Ingest → Preprocess → Classify → Extract → Summarize) for news analysis
  - **Routing**: Automatic routing to specialist analyzers (Earnings, Technical, News, General)
  - **Evaluator-Optimizer**: Iterative quality refinement for high-quality analysis
- **Comprehensive Data**: Integrates Yahoo Finance and Alpha Vantage with 20+ technical indicators
- **LangGraph Workflows**: Advanced workflow orchestration with 7 intelligent tools
- **Learning Memory**: Persistent memory system that learns from past analyses
- **Technical Analysis**: Real-time calculation of RSI, MACD, SMA, and other indicators
- **News Integration**: Sentiment analysis of recent news and market impact
- **Rich Visualization**: Beautiful console output with progress bars and formatted tables
- **Intelligent Query Handler**: Single entry point that automatically selects the best workflow

## Architecture

The system is built with a modular architecture that separates concerns and enables easy extension:

```
┌──────────────────────────────────────────────────────────────────┐
│                   Investment Research Agent                      │
├──────────────────────────────────────────────────────────────────┤
│  🆕 Natural Language Interface:                                  │
│  └── extract_stock_symbol() → LLM + Web Search Extraction       │
│                                                                   │
│  Main Query Interface:                                           │
│  └── agent.query() → Intelligent Workflow Routing               │
│                                                                   │
│  Agentic Workflows:                                              │
│  ├── NewsChainWorkflow        (Prompt Chaining)                 │
│  ├── RoutingWorkflow          (Specialist Routing)              │
│  └── EvaluatorOptimizer       (Quality Refinement)              │
│                                                                   │
│  Core Components:                                                │
│  ├── InvestmentAgent          (Main Orchestrator)               │
│  ├── YahooFinanceDataWrapper  (Primary Data Source)             │
│  ├── AlphaVantageDataWrapper  🆕 (Fallback Data Source)         │
│  ├── WebToolsWrapper          (Web Search & Calculator)         │
│  ├── Analyzer                 (AI Analysis Engine)              │
│  ├── MemorySystem             (Learning & Memory)               │
│  └── LangGraph Integration    (7 Tools: Yahoo + AV + Web)       │
└──────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before running the project, ensure you have:

- **Python 3.9.6** installed
- **OpenAI API Key** (required for AI analysis)
- **Alpha Vantage API Key** (optional but recommended for fallback data)
- **Tavily API Key** (optional, for natural language query enhancement)
- **Internet connection** (for fetching financial data)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API keys**:
   Create a `.env` file in the project directory:
   ```bash
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional but recommended
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   TAVILY_API_KEY=your_tavily_key
   ```
   Or set as environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   export TAVILY_API_KEY=your_tavily_key
   ```
   
   **Getting API Keys**:
   - **OpenAI**: https://platform.openai.com/api-keys (Required)
   - **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (Free tier: 25 requests/day)
   - **Tavily**: https://tavily.com (For natural language query enhancement)

## 🚀 Quick Start

### Basic Usage (Recommended)

```python
from investment_agent import InvestmentAgent

# Initialize the agent
agent = InvestmentAgent()

# Ask any question - the agent automatically selects the best workflow
result = agent.query(
    stock_symbol="AAPL",
    user_query="Should I buy this stock?",
    use_optimizer=True  # Enable quality optimization
)

# Access the analysis
print(f"Workflow used: {result['workflow_used']}")
print(f"Analysis: {result['final_analysis']}")
```

### 🆕 Natural Language Query (New Feature!)

```python
from investment_agent import InvestmentAgent

agent = InvestmentAgent()

# Ask in plain English - no need to know the ticker symbol!
extraction = agent.extract_stock_symbol("What's the latest on Apple stock?")

if extraction['stock_symbol']:
    print(f"Found stock: {extraction['stock_symbol']}")  # Output: AAPL
    
    # Now analyze it
    result = agent.query(
        extraction['stock_symbol'],
        "Give me a comprehensive analysis"
    )
```

**Supported Natural Language Formats**:
- "What's the latest on Apple?"
- "Should I invest in Tesla?"
- "Tell me about Microsoft's performance"
- "Is Google a good buy right now?"
- "Analyze Amazon stock for me"

### Different Query Examples

```python
# News sentiment analysis (uses Prompt Chaining workflow)
result = agent.query("TSLA", "What's the latest news sentiment?")

# Technical analysis (routes to Technical Specialist)
result = agent.query("NVDA", "What's the price trend?")

# Earnings analysis (routes to Earnings Specialist)
result = agent.query("META", "How are the earnings?")

# Comprehensive analysis (uses Evaluator-Optimizer)
result = agent.query("GOOGL", "Is this a good investment?")
```

### Running the Example

```bash
# Interactive mode with workflow demonstrations
python investment_agent.py

# Or run specific examples
python test_agentic_usage.py
```

This will prompt you for a stock symbol and query, then automatically select and execute the appropriate workflow.

## Data Sources & Analysis

### 🆕 Multi-Source Financial Data Integration

The system now uses **dual data sources** with automatic fallback for maximum reliability:

#### Primary Source: Yahoo Finance
- **Historical Prices**: 6 months of daily/weekly data
- **Financial Metrics**: P/E ratio, market cap, beta, dividend yield
- **Company Information**: Sector, industry, analyst recommendations
- **News Articles**: Latest news with sentiment analysis

#### Fallback Source: Alpha Vantage
- **Real-time Quotes**: Current price, volume, change
- **Company Overview**: Comprehensive fundamentals and metrics
- **News & Sentiment**: Market news with AI sentiment scores
- **Technical Indicators**: RSI, MACD, SMA, and more

#### How the Fallback Works

```
Data Request → Try Yahoo Finance
                      ↓
              Success? → Use Yahoo Data
                      ↓
              Failed? → Try Alpha Vantage
                      ↓
              Success? → Use Alpha Vantage Data
                      ↓
              Failed? → Return empty with error message
```

This ensures **99%+ uptime** for data collection, as the system automatically switches sources when one is unavailable.

### Technical Indicators

Automatically calculates 20+ technical indicators:

- **Momentum**: RSI, Stochastic Oscillator, Williams %R
- **Trend**: SMA, EMA, MACD, ADX
- **Volume**: VWAP, On-Balance Volume
- **Volatility**: Bollinger Bands, Average True Range

### AI Analysis

The AI-powered analyzer provides:

- **Price Trend Analysis**: Bullish/bearish trend identification
- **Valuation Assessment**: P/E ratio interpretation and market positioning
- **News Sentiment**: Impact analysis of recent news
- **Investment Recommendations**: Specific buy/hold/sell recommendations
- **Confidence Scoring**: Analysis reliability assessment

## 🤖 Agentic Workflow Patterns

The agent implements three sophisticated workflow patterns that work together to provide intelligent, high-quality analysis:

### 1. Prompt Chaining (News Analysis)

**Pattern**: Ingest → Preprocess → Classify → Extract → Summarize

A 5-step pipeline for systematic news analysis:

```
Step 1: Ingest       → Collect raw news articles
Step 2: Preprocess   → Clean and structure data
Step 3: Classify     → Categorize by topic and sentiment
Step 4: Extract      → Pull key points and insights
Step 5: Summarize    → Create executive summary
```

**Triggered by**: Queries about news, headlines, announcements, sentiment

### 2. Routing (Specialist Analysis)

**Pattern**: Query → Intelligent Router → Specialist → Expert Analysis

Routes queries to the most appropriate specialist:

- **Earnings Specialist**: Revenue, profit, EPS, financial metrics
- **Technical Specialist**: Price trends, charts, indicators
- **News Specialist**: Sentiment, events, market perception
- **General Specialist**: Comprehensive balanced analysis

**Triggered by**: Specific keywords like "earnings", "price", "technical", etc.

### 3. Evaluator-Optimizer (Quality Refinement)

**Pattern**: Generate → Evaluate → Refine → Repeat

Iteratively improves analysis quality:

```
Iteration 1: Generate initial analysis → Evaluate quality
             If score < 8.0 → Identify issues → Refine

Iteration 2: Generate improved analysis → Evaluate quality
             If score ≥ 8.0 → Return final result
```

**Quality Metrics**: Completeness, Accuracy, Actionability, Clarity

**Always active** when `use_optimizer=True` (recommended for important decisions)

### How Workflow Selection Works

The agent automatically selects the best workflow based on your query:

| Query Keywords | Workflow | Specialist (if routing) |
|----------------|----------|------------------------|
| news, headlines, sentiment | Prompt Chaining | - |
| earnings, revenue, profit | Routing | Earnings |
| price, technical, chart | Routing | Technical |
| general questions | Comprehensive | General |

**Example**:
```python
# Automatically uses News Chain workflow
agent.query("AAPL", "What's the news sentiment?")

# Automatically routes to Earnings Specialist
agent.query("TSLA", "How are earnings looking?")

# Automatically uses comprehensive analysis + optimizer
agent.query("GOOGL", "Should I invest?")
```

For more details, see [agentic_workflows.md](agentic_workflows.md)

## 🔧 Core Components

### 1. InvestmentAgent

The main agent class that orchestrates all analysis:

```python
class InvestmentAgent:
    def __init__(self):
        # Initializes all components
        self.data_wrapper = YahooFinanceDataWrapper()
        self.analyzer = Analyzer()
        self.memory_system = MemorySystem()
        self.graph = self._build_graph()  # LangGraph workflow
```

**Key Methods**:
- `execute_analysis()`: Traditional analysis workflow
- `analyze_stock()`: LangGraph-based analysis
- `get_analysis_summary()`: Memory-based insights

### 2. YahooFinanceDataWrapper

Handles primary data collection from Yahoo Finance:

```python
def fetch_historical_prices(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
    # Fetches historical data with technical indicators
    # Returns structured data with price history and calculated indicators
```

**Features**:
- Real-time data fetching with progress indicators
- Automatic technical indicator calculation
- Error handling and data validation
- Multiple time periods and intervals

### 3. 🆕 AlphaVantageDataWrapper

Provides fallback data collection from Alpha Vantage:

```python
def fetch_quote(self, symbol: str) -> Dict[str, Any]:
    # Fetches real-time quote data
    # Used when Yahoo Finance is unavailable
    
def fetch_company_overview(self, symbol: str) -> Dict[str, Any]:
    # Fetches comprehensive company fundamentals
    
def fetch_market_news_sentiment(self, tickers: str) -> Dict[str, Any]:
    # Fetches news with AI sentiment analysis
```

**Features**:
- Automatic fallback when Yahoo Finance fails
- Sentiment-analyzed news articles
- Comprehensive fundamental data
- Technical indicator support

### 4. Analyzer

AI-powered analysis engine using OpenAI:

```python
def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # Performs comprehensive AI analysis
    # Combines traditional analysis with AI insights
    # Generates actionable recommendations
```

**Analysis Types**:
- Price trend analysis
- Financial health assessment
- News sentiment evaluation
- Investment recommendation generation

### 5. MemorySystem

Persistent learning system that stores and retrieves analysis insights:

```python
def add_memory(self, stock_symbol: str, memory_type: str, content: str):
    # Stores analysis insights for future reference
    # Enables pattern recognition and learning
```

**Memory Features**:
- Persistent JSON storage
- Importance scoring system
- Tag-based categorization
- Symbol-specific memory retrieval

## 🔄 Analysis Workflows

### 1.  Analysis Workflow

```
Stock Symbol → Data Collection → AI Analysis → Memory Storage → Results
```

**Process**:
1. Fetch historical prices and technical indicators
2. Collect financial metrics and company information
3. Gather recent news articles
4. Perform AI-powered comprehensive analysis
5. Store high-confidence insights in memory
6. Return structured analysis results

### 2. LangGraph Analysis Workflow

```
User Question → LangGraph State → Tool Calls → Analysis → Recommendations
```

**Process**:
1. Initialize LangGraph state with user question
2. Route to analyst node for processing
3. Execute tool calls for data collection
4. Generate AI-powered analysis
5. Return structured recommendations

## 📈 Example Output

### Analysis Results

```json
{
  "stock_symbol": "META",
  "analysis_result": {
    "analysis_type": "comprehensive",
    "confidence_score": 0.8,
    "ai_analysis": "Based on the analysis...",
    "recommendations": [
      "Positive price trend - consider long position",
      "Reasonable valuation - good entry point"
    ],
    "findings": {
      "price_analysis": {
        "trend_direction": "bullish",
        "volatility": 0.25,
        "current_price": 150.25
      },
      "financial_analysis": {
        "valuation": "reasonable",
        "market_cap": 7000000000000,
        "recommendation": "buy"
      }
    }
  }
}
```

### Console Visualization

The system provides rich console output with:
- Progress bars with spinners
- Color-coded status indicators
- Formatted tables and panels
- Real-time analysis updates

## Memory System

The learning memory system enables the agent to:

- **Store Analysis Insights**: High-confidence analyses are automatically stored
- **Pattern Recognition**: Identifies recurring patterns across analyses
- **Learning Enhancement**: Improves recommendations based on historical data
- **Context Awareness**: Retrieves relevant past analyses for current stocks

### Memory Types

- `analysis`: Comprehensive analysis results
- `pattern`: Identified market patterns
- `recommendation`: Investment recommendations
- `warning`: Risk alerts and warnings

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional but Recommended
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key    # Fallback data source
TAVILY_API_KEY=your_tavily_key                  # Natural language query enhancement
```

### LangGraph Tools

The agent now includes **7 intelligent tools** for data collection:

#### Yahoo Finance Tools (Primary)
1. `get_stock_prices(ticker)` - Historical prices and technical indicators
2. `get_financial_metrics(ticker)` - Key financial ratios and metrics

#### 🆕 Alpha Vantage Tools (Fallback)
3. `get_alpha_vantage_quote(ticker)` - Real-time quote data
4. `get_alpha_vantage_overview(ticker)` - Company fundamentals
5. `get_alpha_vantage_news(tickers)` - News with sentiment analysis

#### Web & Utility Tools
6. `search_web(query)` - Real-time web search
7. `calculate(expression)` - Financial calculations

The LLM agent automatically selects the most appropriate tools based on the query and data availability.


## Key Technical Achievements

This project showcases several advanced technical concepts:

1. **Agentic Workflow Patterns**: Implementation of three sophisticated patterns:
   - **Prompt Chaining**: Multi-step pipeline for systematic processing
   - **Routing**: Intelligent query routing to specialized analyzers
   - **Evaluator-Optimizer**: Iterative quality refinement loop
2. **🆕 Natural Language Understanding**: Extract stock symbols from conversational queries using LLM + web search
3. **🆕 Multi-Source Data Integration**: Automatic fallback between Yahoo Finance and Alpha Vantage for 99%+ uptime
4. **Intelligent Query Understanding**: Automatic workflow selection based on query analysis
5. **LangGraph Integration**: Advanced workflow orchestration with 7 intelligent tools
6. **AI-Powered Analysis**: Sophisticated prompt engineering and response parsing
7. **Technical Analysis**: Real-time calculation of 20+ complex financial indicators
8. **Memory Systems**: Persistent learning and pattern recognition with importance scoring
9. **Rich Visualization**: Professional console output with progress tracking and panels
10. **Error Resilience**: Comprehensive error handling and graceful degradation
11. **Modular Architecture**: Clean separation of concerns and extensible design
12. **Quality Assurance**: Self-evaluating system with iterative improvement

The codebase demonstrates production-ready patterns for agentic AI applications in financial analysis, making it an excellent reference for similar projects.

## Additional Resources

- **[NEW_FEATURES.md](NEW_FEATURES.md)** 🆕 - Complete guide to natural language queries and Alpha Vantage integration
- **[agentic_workflows.md](agentic_workflows.md)** - Detailed documentation on workflow patterns
- **[test_agentic_usage.py](test_agentic_usage.py)** - 8 examples demonstrating different workflows
- **[investment_agent.py](investment_agent.py)** - Main implementation (2300+ lines)

## What's New in v2.0

### 🎯 Natural Language Query Support
Ask questions naturally without knowing ticker symbols:
- "What's the latest on Apple stock?" → Automatically extracts AAPL
- Uses LLM extraction with web search fallback
- High confidence scoring and user confirmation

### 🔄 Multi-Source Data Integration
Automatic fallback between data sources:
- Primary: Yahoo Finance (free, no API key needed)
- Fallback: Alpha Vantage (optional API key)
- 99%+ uptime guarantee for data collection

### 🛠️ 3 New LangGraph Tools
- `get_alpha_vantage_quote()` - Real-time quotes
- `get_alpha_vantage_overview()` - Company fundamentals  
- `get_alpha_vantage_news()` - News with sentiment

See [NEW_FEATURES.md](NEW_FEATURES.md) for complete documentation and examples.

## Contributing

This is a comprehensive test of agentic AI patterns. Feel free to extend it with:
- Additional specialist analyzers
- New workflow patterns (e.g., ReAct, Tree of Thoughts)
- More data sources (Financial Modeling Prep, IEX Cloud, etc.)
- Enhanced memory features (vector storage, semantic search)
- Voice integration for queries

## License

This project is provided as-is for educational and research purposes.