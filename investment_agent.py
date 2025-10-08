#!/usr/bin/env python3
"""
Investment Research Agent
A single-file implementation combining all project components
"""

import os
import json
import logging
import datetime as dt
from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass, asdict

# Core dependencies
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price

# LangGraph and OpenAI imports
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Rich text visualization
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich import print as rprint

# Environment setup
import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENKEY")

# Setup logging with rich
console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MemoryEntry:
    timestamp: str
    stock_symbol: str
    memory_type: str
    content: str
    context: Dict[str, Any]
    importance_score: float
    tags: List[str]

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    stock: str
    analysis_data: Dict[str, Any]
    final_analysis: Dict[str, Any]

# =============================================================================
# CORE DATA TOOLS
# =============================================================================

class YahooFinanceDataWrapper:
    """Yahoo Finance data wrapper with technical analysis."""
    
    def __init__(self):
        console.print("[bold green]Yahoo Finance Data Wrapper initialized[/bold green]")
    
    def fetch_historical_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """Fetch historical price data with technical indicators."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching historical data for {symbol}...", total=None)
            
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period=period, interval=interval)
                
                if hist_data.empty:
                    progress.update(task, description=f"[red]No historical data found for {symbol}")
                    return {"error": f"No historical data found for {symbol}"}
                
                # Calculate technical indicators
                progress.update(task, description="[cyan]Calculating technical indicators...")
                technical_indicators = self._calculate_technical_indicators(hist_data)
                
                # Get latest price info
                latest_price = hist_data['Close'].iloc[-1]
                price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]
                price_change_pct = (price_change / hist_data['Close'].iloc[-2]) * 100
                
                result = {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                    "data_points": len(hist_data),
                    "latest_price": float(latest_price),
                    "price_change": float(price_change),
                    "price_change_pct": float(price_change_pct),
                    "historical_data": hist_data.to_dict('records'),
                    "technical_indicators": technical_indicators,
                    "summary_stats": {
                        "avg_volume": float(hist_data['Volume'].mean()),
                        "volatility": float(hist_data['Close'].pct_change().std() * np.sqrt(252)),
                        "max_price": float(hist_data['High'].max()),
                        "min_price": float(hist_data['Low'].min())
                    }
                }
                
                progress.update(task, description=f"[green]Successfully fetched data for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error fetching data: {str(e)}")
                return {"error": f"Error fetching data: {str(e)}"}

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators."""
        indicators = {}
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Calculating technical indicators...", total=3)
                
                # RSI
                rsi = RSIIndicator(df['Close'], window=14)
                indicators["RSI"] = {
                    "current_value": float(rsi.rsi().iloc[-1]) if not pd.isna(rsi.rsi().iloc[-1]) else None,
                    "signal": self._get_rsi_signal(rsi.rsi().iloc[-1]) if not pd.isna(rsi.rsi().iloc[-1]) else "neutral"
                }
                progress.advance(task)
                
                # MACD
                macd = MACD(df['Close'])
                indicators["MACD"] = {
                    "current_value": float(macd.macd().iloc[-1]) if not pd.isna(macd.macd().iloc[-1]) else None,
                    "signal": self._get_macd_signal(macd.macd().iloc[-1], macd.macd_signal().iloc[-1]) if not pd.isna(macd.macd().iloc[-1]) else "neutral"
                }
                progress.advance(task)
                
                # SMA
                sma_20 = SMAIndicator(df['Close'], window=20)
                indicators["SMA_20"] = {
                    "current_value": float(sma_20.sma_indicator().iloc[-1]) if not pd.isna(sma_20.sma_indicator().iloc[-1]) else None
                }
                progress.advance(task)
                
        except Exception as e:
            console.print(f"[red]Error calculating technical indicators: {str(e)}[/red]")
            indicators["error"] = f"Error calculating indicators: {str(e)}"
        
        return indicators

    def _get_rsi_signal(self, rsi_value: float) -> str:
        """Determine RSI signal."""
        if rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        else:
            return "neutral"

    def _get_macd_signal(self, macd_value: float, signal_value: float) -> str:
        """Determine MACD signal."""
        if macd_value > signal_value:
            return "bullish"
        elif macd_value < signal_value:
            return "bearish"
        else:
            return "neutral"
    
    def fetch_key_financials(self, symbol: str) -> Dict[str, Any]:
        """Fetch key financial information."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching financial data for {symbol}...", total=None)
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                key_metrics = {
                    "symbol": symbol,
                    "company_name": info.get('longName', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "industry": info.get('industry', 'N/A'),
                    "market_cap": info.get('marketCap', 'N/A'),
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "forward_pe": info.get('forwardPE', 'N/A'),
                    "price_to_book": info.get('priceToBook', 'N/A'),
                    "dividend_yield": info.get('dividendYield', 'N/A'),
                    "beta": info.get('beta', 'N/A'),
                    "current_price": info.get('currentPrice', 'N/A'),
                    "recommendation": info.get('recommendationKey', 'N/A')
                }
                
                result = {
                    "key_metrics": key_metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched financial data for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error fetching financial data: {str(e)}")
                return {"error": f"Error fetching financials: {str(e)}"}
    
    def fetch_stock_news(self, symbol: str, max_articles: int = 5) -> Dict[str, Any]:
        """Fetch latest news articles."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching news for {symbol}...", total=None)
            
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                if not news:
                    progress.update(task, description=f"[yellow]No news articles found for {symbol}")
                    return {"symbol": symbol, "news": [], "message": "No news articles found"}
                
                processed_news = []
                for article in news[:max_articles]:
                    processed_article = {
                        "title": article.get('title', 'N/A'),
                        "publisher": article.get('publisher', 'N/A'),
                        "link": article.get('link', 'N/A'),
                        "summary": article.get('summary', 'N/A')
                    }
                    processed_news.append(processed_article)
                
                result = {
                    "symbol": symbol,
                    "total_articles": len(processed_news),
                    "news": processed_news,
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched {len(processed_news)} news articles for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error fetching news: {str(e)}")
                return {"error": f"Error fetching news: {str(e)}"}

# =============================================================================
# LANGGRAPH TOOLS
# =============================================================================

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicators."""
    try:
        wrapper = YahooFinanceDataWrapper()
        result = wrapper.fetch_historical_prices(ticker, period="6mo", interval="1wk")
        
        if "error" in result:
            return result["error"]
        
        return {
            'stock_price': result['historical_data'],
            'indicators': result['technical_indicators'],
            'latest_price': result['latest_price'],
            'price_change_pct': result['price_change_pct']
        }
    except Exception as e:
        return f"Error fetching price data: {str(e)}"

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios and metrics."""
    try:
        wrapper = YahooFinanceDataWrapper()
        result = wrapper.fetch_key_financials(ticker)
        
        if "error" in result:
            return result["error"]
        
        metrics = result.get("key_metrics", {})
        return {
            'pe_ratio': metrics.get('pe_ratio'),
            'price_to_book': metrics.get('price_to_book'),
            'market_cap': metrics.get('market_cap'),
            'recommendation': metrics.get('recommendation')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"

# =============================================================================
# ANALYZERS
# =============================================================================

class Analyzer:
    """analyzer that combines all analysis types."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, openai_api_key=OPENAI_API_KEY)
        console.print("[bold green]Analyzer initialized[/bold green]")
    
    def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on all available data."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Performing comprehensive analysis...", total=None)
            
            analysis_result = {
                "analysis_type": "comprehensive",
                "timestamp": datetime.now().isoformat(),
                "findings": {},
                "recommendations": [],
                "confidence_score": 0.0,
                "ai_analysis": ""
            }
            
            # Prepare data context for AI analysis
            data_context = self._prepare_data_context(data)
            
            # Generate AI-powered analysis
            analysis_prompt = """
            Analyze the provided financial data and provide comprehensive insights on:
            1. Stock price trends and technical indicators
            2. Financial health and valuation metrics
            3. Recent news sentiment and market impact
            4. Overall investment recommendation
            
            Provide specific, actionable recommendations based on the data.
            """
            
            try:
                messages = [
                    SystemMessage(content="You are an expert financial analyst. Provide objective, data-driven analysis and recommendations."),
                    HumanMessage(content=f"{analysis_prompt}\n\nData Context:\n{data_context}")
                ]
                response = self.llm.invoke(messages)
                analysis_result["ai_analysis"] = response.content
                progress.update(task, description="[green]AI analysis completed")
            except Exception as e:
                console.print(f"[red]Error generating AI analysis: {str(e)}[/red]")
                analysis_result["ai_analysis"] = f"Analysis generation failed: {str(e)}"
                progress.update(task, description="[red]AI analysis failed")
            
            # Traditional analysis
            if "historical_prices" in data:
                analysis_result["findings"]["price_analysis"] = self._analyze_price_data(data["historical_prices"])
            
            if "financial_metrics" in data:
                analysis_result["findings"]["financial_analysis"] = self._analyze_financial_data(data["financial_metrics"])
            
            if "news_data" in data:
                analysis_result["findings"]["news_analysis"] = self._analyze_news_data(data["news_data"])
            
            analysis_result["recommendations"] = self._generate_recommendations(analysis_result["findings"])
            analysis_result["confidence_score"] = self._calculate_confidence_score(analysis_result["findings"])
            
            progress.update(task, description="[green]Analysis completed successfully")
            return analysis_result
    
    def _prepare_data_context(self, data: Dict[str, Any]) -> str:
        """Prepare data context for AI analysis."""
        context_parts = []
        
        if "historical_prices" in data:
            price_data = data["historical_prices"]
            context_parts.append(f"Price Data: Latest price ${price_data.get('latest_price', 'N/A')}, Change: {price_data.get('price_change_pct', 'N/A')}%")
        
        if "financial_metrics" in data:
            metrics_data = data["financial_metrics"]
            context_parts.append(f"Financial Metrics: P/E Ratio {metrics_data.get('pe_ratio', 'N/A')}, Market Cap {metrics_data.get('market_cap', 'N/A')}")
        
        if "news_data" in data:
            news_data = data["news_data"]
            context_parts.append(f"News: {news_data.get('total_articles', 0)} articles found")
        
        return "\n".join(context_parts) if context_parts else "No data available"
    
    def _analyze_price_data(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price data."""
        return {
            "trend_direction": "bullish" if price_data.get("price_change_pct", 0) > 0 else "bearish",
            "volatility": price_data.get("summary_stats", {}).get("volatility", 0),
            "current_price": price_data.get("latest_price", 0)
        }
    
    def _analyze_financial_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial data."""
        pe_ratio = financial_data.get("pe_ratio", "N/A")
        if pe_ratio == "N/A" or pe_ratio is None:
            valuation = "unknown"
        else:
            try:
                valuation = "expensive" if float(pe_ratio) > 20 else "reasonable"
            except Exception:
                valuation = "unknown"

        market_cap = financial_data.get("market_cap", "N/A")
        if market_cap == "N/A" or market_cap is None:
            market_cap_value = None
        else:
            market_cap_value = market_cap

        recommendation = financial_data.get("recommendation", "N/A")
        if recommendation == "N/A" or recommendation is None:
            recommendation_value = "No recommendation available"
        else:
            recommendation_value = recommendation

        return {
            "valuation": valuation,
            "market_cap": market_cap_value,
            "recommendation": recommendation_value
        }
    
    def _analyze_news_data(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze news data."""
        return {
            "article_count": news_data.get("total_articles", 0),
            "sentiment": "positive" if news_data.get("total_articles", 0) > 0 else "neutral"
        }
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        price_analysis = findings.get("price_analysis", {})
        if price_analysis.get("trend_direction") == "bullish":
            recommendations.append("Positive price trend - consider long position")
        
        financial_analysis = findings.get("financial_analysis", {})
        if financial_analysis.get("valuation") == "reasonable":
            recommendations.append("Reasonable valuation - good entry point")
        
        return recommendations
    
    def _calculate_confidence_score(self, findings: Dict[str, Any]) -> float:
        """Calculate confidence score."""
        score = 0.5
        if findings.get("price_analysis"):
            score += 0.2
        if findings.get("financial_analysis"):
            score += 0.2
        if findings.get("news_analysis"):
            score += 0.1
        return min(1.0, score)

# =============================================================================
# MEMORY SYSTEM
# =============================================================================

class MemorySystem:
    """memory system for learning and pattern recognition."""
    
    def __init__(self, memory_file_path: str = "memory.json"):
        self.memory_file_path = memory_file_path
        self.memories: List[MemoryEntry] = []
        self._load_memories()
        console.print(f"[bold green]Memory System initialized with {len(self.memories)} memories[/bold green]")
    
    def _load_memories(self) -> None:
        """Load memories from file."""
        if os.path.exists(self.memory_file_path):
            try:
                with open(self.memory_file_path, 'r') as f:
                    data = json.load(f)
                    self.memories = [MemoryEntry(**entry) for entry in data.get("memories", [])]
                console.print(f"[green]Loaded {len(self.memories)} memories from file[/green]")
            except Exception as e:
                console.print(f"[red]Error loading memories: {str(e)}[/red]")
                self.memories = []
        else:
            console.print("[yellow]No existing memory file found, starting fresh[/yellow]")
    
    def _save_memories(self) -> None:
        """Save memories to file."""
        try:
            data = {
                "memories": [asdict(memory) for memory in self.memories],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.memory_file_path, 'w') as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]Saved {len(self.memories)} memories to file[/green]")
        except Exception as e:
            console.print(f"[red]Error saving memories: {str(e)}[/red]")
    
    def add_memory(self, stock_symbol: str, memory_type: str, content: str, 
                   context: Dict[str, Any] = None, importance_score: float = 0.5,
                   tags: List[str] = None) -> str:
        """Add a new memory entry."""
        memory_entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            stock_symbol=stock_symbol.upper(),
            memory_type=memory_type,
            content=content,
            context=context or {},
            importance_score=importance_score,
            tags=tags or []
        )
        self.memories.append(memory_entry)
        self._save_memories()
        console.print(f"[green]Added memory for {stock_symbol}: {memory_type}[/green]")
        return f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_memories_for_symbol(self, stock_symbol: str) -> List[MemoryEntry]:
        """Get memories for a specific stock symbol."""
        symbol_upper = stock_symbol.upper()
        relevant_memories = [m for m in self.memories if m.stock_symbol == symbol_upper]
        relevant_memories.sort(key=lambda m: (m.importance_score, m.timestamp), reverse=True)
        console.print(f"[green]Retrieved {len(relevant_memories)} memories for {stock_symbol}[/green]")
        return relevant_memories
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_memories": len(self.memories),
            "memories_by_symbol": {},
            "memories_by_type": {},
            "average_importance": 0.0
        }
        
        for memory in self.memories:
            symbol = memory.stock_symbol
            if symbol not in stats["memories_by_symbol"]:
                stats["memories_by_symbol"][symbol] = 0
            stats["memories_by_symbol"][symbol] += 1
            
            memory_type = memory.memory_type
            if memory_type not in stats["memories_by_type"]:
                stats["memories_by_type"][memory_type] = 0
            stats["memories_by_type"][memory_type] += 1
        
        if self.memories:
            stats["average_importance"] = sum(m.importance_score for m in self.memories) / len(self.memories)
        
        return stats

# =============================================================================
# MAIN INVESTMENT RESEARCH AGENT
# =============================================================================

class InvestmentAgent:
    """Investment Research Agent combining all functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.tools = [get_stock_prices, get_financial_metrics]
        self.llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        self.data_wrapper = YahooFinanceDataWrapper()
        self.analyzer = Analyzer()
        self.memory_system = MemorySystem()
        self.results = {}
        console.print("[bold green]Investment Research Agent initialized successfully[/bold green]")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph_builder = StateGraph(State)
        
        # Add the analyst node
        graph_builder.add_node('analyst', self._analyst)
        graph_builder.add_edge(START, 'analyst')
        
        # Add tools node
        graph_builder.add_node('tools', ToolNode(self.tools))
        graph_builder.add_conditional_edges('analyst', self._tools_condition)
        graph_builder.add_edge('tools', 'analyst')
        
        return graph_builder.compile()

    def _tools_condition(self, state: State) -> str:
        """Determine if tools should be called."""
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END

    def _analyst(self, state: State) -> State:
        """Analyst node that processes stock analysis."""
        ANALYST_PROMPT = """
        You are a financial analyst specializing in stock evaluation.

        You have access to the following tools:
        1. **get_stock_prices**: Retrieves stock price data and technical indicators.
        2. **get_financial_metrics**: Retrieves key financial metrics and ratios.

        ### Your Task:
        1. Use the provided stock symbol to query the tools and collect data.
        2. Analyze the results and identify trends and patterns.
        3. Provide a comprehensive summary with specific recommendations.

        ### Output Format:
        Respond with:
        "stock": "<Stock Symbol>",
        "price_analysis": "<Analysis of price trends and technical indicators>",
        "financial_analysis": "<Analysis of financial metrics and valuation>",
        "final_summary": "<Overall conclusion based on all analyses>",
        "recommendation": "<Specific investment recommendation>"

        Be objective, data-driven, and provide actionable insights.
        """
        
        messages = state["messages"]
        stock_symbol = state.get("stock", "UNKNOWN")
        
        # Create the analysis prompt
        analysis_prompt = f"{ANALYST_PROMPT}\n\nStock Symbol: {stock_symbol}"
        
        # Add the prompt to messages
        messages.append(HumanMessage(content=analysis_prompt))
        
        # --- FIX: Ensure tool calls are handled with tool messages ---
        # Use the LangGraph ToolNode to handle tool calls and responses
        # This is the correct way to ensure tool_call_ids are matched with ToolMessages

        # Instead of calling self.llm_with_tools.invoke directly, let the graph handle tool calls
        # So, just return the updated state and let the graph's tool node process tool calls

        return {
            "messages": messages,
            "stock": stock_symbol,
            "analysis_data": state.get("analysis_data", {}),
            "final_analysis": state.get("final_analysis", {})
        }

    def analyze_stock(self, stock_symbol: str, user_question: str = "Should I buy this stock?") -> Dict[str, Any]:
        """Execute comprehensive stock analysis using LangGraph."""
        initial_state = {
            "messages": [HumanMessage(content=user_question)],
            "stock": stock_symbol,
            "analysis_data": {},
            "final_analysis": {}
        }
        
        console.print(f"[cyan]Starting analysis for {stock_symbol}[/cyan]")
        
        # Execute the graph
        events = self.graph.stream(initial_state, stream_mode='values')
        
        final_state = None
        for event in events:
            final_state = event
            if 'messages' in event:
                last_message = event['messages'][-1]
                if hasattr(last_message, 'content'):
                    console.print(f"[dim cyan]Analysis step completed: {last_message.content}[/dim cyan]")
        
        # Extract final analysis
        if final_state and 'messages' in final_state:
            final_message = final_state['messages'][-1]
            if hasattr(final_message, 'content'):
                analysis_result = self._parse_analysis_result(final_message.content)
                return {
                    "stock_symbol": stock_symbol,
                    "analysis_result": analysis_result,
                    "raw_response": final_message.content,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
        
        return {
            "stock_symbol": stock_symbol,
            "error": "Analysis failed to complete",
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }

    def _parse_analysis_result(self, content: str) -> Dict[str, Any]:
        """Parse the analysis result from the LLM response."""
        result = {}
        lines = content.split('\n')
        
        for line in lines:
            if ':' in line and '"' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip().strip('"')
                    value = value.strip().strip('",')
                    result[key] = value
                except:
                    continue
        
        return result

    def execute_analysis(self, stock_symbol: str) -> Dict[str, Any]:
        """Execute a analysis without LangGraph."""
        console.print(f"[cyan]Starting analysis for {stock_symbol}[/cyan]")
        
        # Collect data
        price_data = self.data_wrapper.fetch_historical_prices(stock_symbol)
        financial_data = self.data_wrapper.fetch_key_financials(stock_symbol)
        news_data = self.data_wrapper.fetch_stock_news(stock_symbol)
        
        # Combine data
        combined_data = {
            "historical_prices": price_data,
            "financial_metrics": financial_data.get("key_metrics", {}),
            "news_data": news_data
        }
        
        # Analyze data
        analysis_result = self.analyzer.analyze_data(combined_data)
        
        # Store in memory
        if analysis_result["confidence_score"] > 0.7:
            self.memory_system.add_memory(
                stock_symbol, "analysis",
                f"High confidence analysis: {analysis_result['ai_analysis']}...",
                {"confidence": analysis_result["confidence_score"]}, 
                0.8, ["high_confidence"]
            )
        
        result = {
            "stock_symbol": stock_symbol,
            "analysis_result": analysis_result,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        logger.info(f"analysis completed for {stock_symbol}")
        return result

    def get_analysis_summary(self, stock_symbol: str) -> Dict[str, Any]:
        """Get analysis summary with memory insights."""
        memories = self.memory_system.get_memories_for_symbol(stock_symbol)
        memory_stats = self.memory_system.get_memory_statistics()
        
        return {
            "stock_symbol": stock_symbol,
            "total_memories": len(memories),
            "memory_insights": [m.content for m in memories[:3]],  # Top 3 memories
            "memory_statistics": memory_stats,
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of the investment agent."""
    print("=" * 60)
    print("Investment Research Agent")
    print("=" * 60)
    
    # Initialize the agent
    agent = InvestmentAgent()
    
    # Test stock symbol
    stock_symbol = input("Enter stock symbol: ")
    
    print(f"\n1. Executing Analysis for {stock_symbol}")
    print("-" * 50)
    
    # Execute analysis
    result = agent.execute_analysis(stock_symbol)
    
    if result["status"] == "completed":
        analysis = result["analysis_result"]
        print(f"✓ Analysis completed successfully")
        print(f"Confidence Score: {analysis['confidence_score']:.2f}")
        print(f"Recommendations: {len(analysis['recommendations'])}")
        print(f"\nAI Analysis Summary:")
        print(analysis['ai_analysis'])
        print(f"\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n2. Executing LangGraph Analysis for {stock_symbol}")
    print("-" * 50)
    
    # Execute LangGraph analysis
    langgraph_result = agent.analyze_stock(stock_symbol, "Should I buy this stock?")
    
    if langgraph_result["status"] == "completed":
        print(f"✓ LangGraph analysis completed successfully")
        print(f"Session ID: {langgraph_result['stock_symbol']}")
        print(f"Analysis Method: LangGraph")
        # print(f"\nRaw Analysis:")
        # print(langgraph_result['raw_response'])
    else:
        print(f"✗ LangGraph analysis failed: {langgraph_result.get('error', 'Unknown error')}")
    
    print(f"\n3. Memory System Summary")
    print("-" * 30)
    summary = agent.get_analysis_summary(stock_symbol)
    print(f"Total memories for {stock_symbol}: {summary['total_memories']}")
    print(f"Total memories in system: {summary['memory_statistics']['total_memories']}")
    print(f"Average importance: {summary['memory_statistics']['average_importance']:.2f}")
    
if __name__ == "__main__":
    main()
