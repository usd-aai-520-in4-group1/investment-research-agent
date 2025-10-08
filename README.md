# Investment Research Agent

A sophisticated AI-powered investment research system that combines LangGraph workflows, OpenAI's GPT models, and comprehensive financial data analysis to provide intelligent stock recommendations.

## 🚀 Overview

This project is a complete investment research platform that leverages cutting-edge AI technology to analyze stocks comprehensively. It combines traditional financial analysis with modern AI capabilities to deliver actionable investment insights.

### Key Features

- **🤖 AI-Powered Analysis**: Uses OpenAI's GPT-4o-mini for intelligent financial analysis
- **📊 Comprehensive Data**: Integrates Yahoo Finance data with 20+ technical indicators
- **🔄 LangGraph Workflows**: Advanced workflow orchestration for complex analysis chains
- **🧠 Learning Memory**: Persistent memory system that learns from past analyses
- **📈 Technical Analysis**: Real-time calculation of RSI, MACD, SMA, and other indicators
- **📰 News Integration**: Sentiment analysis of recent news and market impact
- **🎨 Rich Visualization**: Beautiful console output with progress bars and formatted tables
- **⚡ Dual Analysis Modes**: Both LangGraph-based and traditional analysis workflows

## 🏗️ Architecture

The system is built with a modular architecture that separates concerns and enables easy extension:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Investment Research Agent                    │
├─────────────────────────────────────────────────────────────────┤
│  Core Components:                                               │
│  ├── InvestmentAgent      (Main Agent Class)              │
│  ├── YahooFinanceDataWrapper     (Data Collection)             │
│  ├── Analyzer             (AI Analysis Engine)            │
│  ├── MemorySystem         (Learning & Memory)            │
│  └── LangGraph Integration      (Workflow Orchestration)       │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

Before running the project, ensure you have:

- **Python 3.9.6** installed
- **OpenAI API Key** (required for AI analysis)
- **Internet connection** (for fetching financial data)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install pandas numpy yfinance ta langgraph langchain-openai langchain-core rich python-dotenv
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the project directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   Or set it as an environment variable:
   ```bash
   export OPENKEY=your_openai_api_key_here
   ```

## 🚀 Quick Start

### Basic Usage

```python
from investment_agent import InvestmentAgent

# Initialize the agent
agent = InvestmentAgent()

# Analyze a stock using  analysis
result = agent.execute_analysis("META")

# Analyze using LangGraph workflow
langgraph_result = agent.analyze_stock("META", "Should I buy META stock?")
```

### Running the Example

```bash
python investment_agent.py
```

This will run a complete analysis example for META stock, demonstrating both analysis modes.

## 📊 Data Sources & Analysis

### Financial Data Integration

The system fetches comprehensive data from Yahoo Finance:

- **Historical Prices**: 6 months of daily/weekly data
- **Financial Metrics**: P/E ratio, market cap, beta, dividend yield
- **Company Information**: Sector, industry, analyst recommendations
- **News Articles**: Latest news with sentiment analysis

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

Handles all data collection from Yahoo Finance:

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

### 3. Analyzer

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

### 4. MemorySystem

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

## 🧠 Memory System

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

# Alternative
OPENKEY=your_openai_api_key
```


## 🎯 Key Technical Achievements

This project showcases several advanced technical concepts:

1. **LangGraph Integration**: Advanced workflow orchestration with state management
2. **AI-Powered Analysis**: Sophisticated prompt engineering and response parsing
3. **Technical Analysis**: Real-time calculation of complex financial indicators
4. **Memory Systems**: Persistent learning and pattern recognition
5. **Rich Visualization**: Professional console output with progress tracking
6. **Error Resilience**: Comprehensive error handling and graceful degradation
7. **Modular Architecture**: Clean separation of concerns and extensible design

The codebase demonstrates production-ready patterns for AI applications in financial analysis, making it an excellent reference for similar projects.