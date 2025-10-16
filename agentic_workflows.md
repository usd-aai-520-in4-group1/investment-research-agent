# Agentic Workflow Patterns

This document explains the three agentic workflow patterns implemented in the Investment Research Agent.

## Overview

The agent uses intelligent workflow routing to provide the best analysis based on your query. It automatically selects from three powerful patterns:

1. **Prompt Chaining** - For news analysis
2. **Routing** - For specialist analysis  
3. **Evaluator-Optimizer** - For quality refinement

## 1. Prompt Chaining Workflow

**Pattern:** Ingest → Preprocess → Classify → Extract → Summarize

**When used:** Automatically triggered for news-related queries containing keywords like:
- "news", "headlines", "announcements", "sentiment", "media"

**How it works:**
```
Step 1: Ingest     → Collect raw news articles
Step 2: Preprocess → Clean and structure the data
Step 3: Classify   → Categorize by topic and sentiment
Step 4: Extract    → Pull out key points and insights
Step 5: Summarize  → Create executive summary
```

**Example queries:**
- "What's the latest news sentiment for AAPL?"
- "Show me recent headlines about Tesla"
- "What are the media announcements for this stock?"

**Benefits:**
- Systematic processing of news data
- Removes noise and focuses on key information
- Provides clear sentiment indicators

## 2. Routing Workflow

**Pattern:** Query → Route to Specialist → Specialized Analysis

**When used:** Automatically triggered for specialist queries containing keywords like:
- **Earnings:** "earnings", "revenue", "profit", "eps"
- **Technical:** "price", "technical", "chart", "trend", "support", "resistance"
- **News:** "news", "announcement", "event", "sentiment"

**Available specialists:**
- **Earnings Specialist** - Deep dive into financial metrics and profitability
- **Technical Specialist** - Price action and technical indicators analysis
- **News Specialist** - Sentiment and event impact analysis
- **General Specialist** - Comprehensive balanced analysis

**How it works:**
```
User Query → Intelligent Router → Appropriate Specialist → Expert Analysis
```

**Example queries:**
- "How are the earnings looking for META?" → Routes to Earnings Specialist
- "What's the technical analysis for NVDA?" → Routes to Technical Specialist
- "Should I buy GOOGL based on price trends?" → Routes to Technical Specialist

**Benefits:**
- Expert-level analysis for specific aspects
- Focused insights without information overload
- Faster processing with targeted approach

## 3. Evaluator-Optimizer Workflow

**Pattern:** Generate → Evaluate → Refine → Repeat

**When used:** Enabled by default (can be disabled with `use_optimizer=False`)

**How it works:**
```
Iteration 1:
  → Generate initial analysis
  → Evaluate quality (completeness, accuracy, actionability, clarity)
  → Score < 8.0? → Refine and improve
  
Iteration 2 (if needed):
  → Generate improved analysis
  → Evaluate quality again
  → Score ≥ 8.0? → Return final analysis
```

**Quality metrics evaluated:**
- **Completeness** - Covers all key aspects
- **Accuracy** - Logical and well-reasoned
- **Actionability** - Clear recommendations
- **Clarity** - Easy to understand

**Benefits:**
- Self-improving analysis quality
- Catches incomplete or unclear responses
- Ensures actionable recommendations

## Using the Agent

### Basic Usage

```python
from investment_agent import InvestmentAgent

# Initialize agent
agent = InvestmentAgent()

# Ask a question (automatic workflow selection)
result = agent.query(
    stock_symbol="AAPL",
    user_query="Should I buy this stock?",
    use_optimizer=True  # Enable quality optimization
)

# Access results
print(f"Workflow used: {result['workflow_used']}")
print(f"Analysis: {result['final_analysis']}")
```

### Query Examples by Workflow

**News Chain Examples:**
```python
# News analysis
result = agent.query("TSLA", "What's the latest news sentiment?")

# Recent announcements
result = agent.query("MSFT", "Show me recent headlines")
```

**Routing Examples:**
```python
# Earnings specialist
result = agent.query("GOOGL", "How are the earnings?")

# Technical specialist
result = agent.query("NVDA", "What's the price trend?")

# General specialist
result = agent.query("AAPL", "Should I invest in this stock?")
```

**With/Without Optimizer:**
```python
# With optimizer (default, higher quality)
result = agent.query("META", "Is this a good buy?", use_optimizer=True)

# Without optimizer (faster)
result = agent.query("AMZN", "Quick price check?", use_optimizer=False)
```

## Command Line Usage

Simply run:
```bash
python investment_agent.py
```

Then follow the prompts:
1. Enter stock symbol (e.g., AAPL)
2. Ask your question
3. Agent automatically selects the best workflow
4. Get comprehensive analysis with workflow details

## Understanding Results

The result contains:

```python
{
    "stock_symbol": "AAPL",
    "user_query": "Should I buy?",
    "workflow_used": "routing",  # Which workflow was selected
    "workflow_details": {
        "workflow": "routing",
        "specialist": "general",  # Which specialist was used
        "optimization": {
            "iterations": 2,
            "final_quality_score": 8.5
        }
    },
    "final_analysis": "...",  # The actual analysis
    "data_summary": {
        "price": 178.45,
        "pe_ratio": 28.5,
        "news_count": 5
    },
    "timestamp": "2025-10-16T...",
    "status": "completed"
}
```

## Workflow Selection Logic

The agent uses keyword matching to determine the workflow:

| Keywords in Query | Selected Workflow | Specialist (if routing) |
|-------------------|-------------------|-------------------------|
| news, headlines, sentiment | News Chain | - |
| earnings, revenue, profit | Routing | Earnings |
| price, technical, chart | Routing | Technical |
| general questions | Comprehensive | General |

## Performance Tips

1. **Be specific** - Clear queries get better routing
2. **Use optimizer for important decisions** - Higher quality analysis
3. **Skip optimizer for quick checks** - Faster response
4. **Combine workflows** - Ask multiple questions to get different perspectives

## Memory Integration

All queries are automatically saved to the memory system with:
- Query text
- Workflow used
- Result summary
- Timestamp

Access memory with:
```python
summary = agent.get_analysis_summary("AAPL")
print(f"Previous analyses: {summary['total_memories']}")
```

## Technical Details

### Architecture

```
InvestmentAgent
├── NewsChainWorkflow (5-step pipeline)
├── RoutingWorkflow (4 specialists)
├── EvaluatorOptimizerWorkflow (iterative refinement)
├── YahooFinanceDataWrapper (price & financials)
├── WebToolsWrapper (web search & calculator)
├── Analyzer (traditional analysis)
└── MemorySystem (learning & recall)
```

### Workflow Classes

- `NewsChainWorkflow` - Implements 5-step prompt chaining
- `RoutingWorkflow` - Routes to 4 different specialists
- `EvaluatorOptimizerWorkflow` - Iterative quality improvement

### Key Methods

- `agent.query()` - Main entry point (automatic routing)
- `agent._determine_workflow()` - Workflow selection logic
- `agent._collect_stock_data()` - Data gathering
- `agent._generate_comprehensive_analysis()` - Default analysis

## Advanced Usage

### Manual Workflow Selection

```python
# Force news chain workflow
news_result = agent.news_chain.execute(
    stock_symbol="AAPL",
    raw_news=news_articles
)

# Force routing to specific specialist
routing_result = agent.routing_workflow.execute(
    stock_symbol="AAPL",
    query="earnings analysis",
    data=stock_data
)

# Manual optimization
optimized = agent.evaluator_optimizer.execute(
    stock_symbol="AAPL",
    initial_analysis=raw_analysis
)
```

### Custom Quality Threshold

Modify `EvaluatorOptimizerWorkflow.max_iterations` or quality threshold:

```python
agent.evaluator_optimizer.max_iterations = 3  # More refinement
```

## Troubleshooting

**Issue:** Wrong workflow selected
- **Solution:** Use more specific keywords in your query

**Issue:** Low quality scores
- **Solution:** Enable optimizer or provide more context

**Issue:** Slow responses
- **Solution:** Disable optimizer for quick queries

**Issue:** Missing data
- **Solution:** Check API keys in .env file

## Examples Output

### News Chain Output
```
→ Step 1/5: Ingesting news...
→ Step 2/5: Preprocessing...
→ Step 3/5: Classifying...
→ Step 4/5: Extracting key points...
→ Step 5/5: Summarizing...
✓ News Chain completed
```

### Routing Output
```
→ Routing to: earnings specialist
→ Earnings specialist analyzing...
```

### Optimizer Output
```
→ Iteration 1/2
Quality Score: 7.20/10
→ Optimizing analysis...
→ Iteration 2/2
Quality Score: 8.50/10
✓ Quality threshold met
```

## Best Practices

1. **Start with general queries** to get comprehensive analysis
2. **Use specialist queries** when you need deep-dive on specific aspects
3. **Enable optimizer** for important investment decisions
4. **Review workflow details** to understand how analysis was generated
5. **Save results** for future reference and pattern recognition

---

*For more information, see the main README.md or examine the source code.*

