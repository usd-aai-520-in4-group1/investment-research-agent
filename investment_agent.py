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
import requests
import time

# Core dependencies
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price
from sympy import sympify

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
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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





class AlphaVantageDataWrapper:
    """Alpha Vantage data wrapper for stock market data, technical indicators, and news."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = None):
        """Initialize Alpha Vantage wrapper with API key."""
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        if not self.api_key:
            console.print("[bold red]Warning: Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY environment variable.[/bold red]")
        else:
            console.print("[bold green]Alpha Vantage Data Wrapper initialized[/bold green]")
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Alpha Vantage."""
        if not self.api_key:
            return {"error": "Alpha Vantage API key not configured"}
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            if "Note" in data:
                return {"error": f"API limit reached: {data['Note']}"}
            
            return data
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse response: {str(e)}"}
    
    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote data for a stock symbol."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching quote for {symbol}...", total=None)
            
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol
                }
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                quote = data.get('Global Quote', {})
                
                if not quote:
                    progress.update(task, description=f"[red]No quote data found for {symbol}")
                    return {"error": f"No quote data found for {symbol}"}
                
                result = {
                    "symbol": quote.get('01. symbol', symbol),
                    "price": float(quote.get('05. price', 0)),
                    "change": float(quote.get('09. change', 0)),
                    "change_percent": quote.get('10. change percent', '0%'),
                    "volume": int(quote.get('06. volume', 0)),
                    "latest_trading_day": quote.get('07. latest trading day', 'N/A'),
                    "previous_close": float(quote.get('08. previous close', 0)),
                    "open": float(quote.get('02. open', 0)),
                    "high": float(quote.get('03. high', 0)),
                    "low": float(quote.get('04. low', 0)),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched quote for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching quote: {str(e)}"}
    
    def fetch_daily_prices(self, symbol: str, outputsize: str = "compact") -> Dict[str, Any]:
        """
        Fetch daily historical prices.
        outputsize: 'compact' (last 100 data points) or 'full' (up to 20 years)
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching daily prices for {symbol}...", total=None)
            
            try:
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'outputsize': outputsize
                }
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                time_series = data.get('Time Series (Daily)', {})
                
                if not time_series:
                    progress.update(task, description=f"[red]No daily data found for {symbol}")
                    return {"error": f"No daily data found for {symbol}"}
                
                # Convert to list of dictionaries
                prices = []
                for date, values in sorted(time_series.items(), reverse=True)[:100]:
                    prices.append({
                        'date': date,
                        'open': float(values.get('1. open', 0)),
                        'high': float(values.get('2. high', 0)),
                        'low': float(values.get('3. low', 0)),
                        'close': float(values.get('4. close', 0)),
                        'volume': int(values.get('5. volume', 0))
                    })
                
                # Calculate basic statistics
                closes = [p['close'] for p in prices]
                volumes = [p['volume'] for p in prices]
                
                result = {
                    "symbol": symbol,
                    "data_points": len(prices),
                    "prices": prices,
                    "latest_price": prices[0]['close'] if prices else 0,
                    "stats": {
                        "avg_close": float(np.mean(closes)) if closes else 0,
                        "avg_volume": float(np.mean(volumes)) if volumes else 0,
                        "max_price": float(max(closes)) if closes else 0,
                        "min_price": float(min(closes)) if closes else 0,
                        "volatility": float(np.std(closes)) if closes else 0
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched {len(prices)} days of data for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching daily prices: {str(e)}"}
    
    def fetch_company_overview(self, symbol: str) -> Dict[str, Any]:
        """Fetch company overview and fundamental data."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching company overview for {symbol}...", total=None)
            
            try:
                params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol
                }
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                if not data or 'Symbol' not in data:
                    progress.update(task, description=f"[red]No overview data found for {symbol}")
                    return {"error": f"No overview data found for {symbol}"}
                
                result = {
                    "symbol": data.get('Symbol', symbol),
                    "company_name": data.get('Name', 'N/A'),
                    "description": data.get('Description', 'N/A'),
                    "sector": data.get('Sector', 'N/A'),
                    "industry": data.get('Industry', 'N/A'),
                    "exchange": data.get('Exchange', 'N/A'),
                    "market_cap": data.get('MarketCapitalization', 'N/A'),
                    "pe_ratio": data.get('PERatio', 'N/A'),
                    "peg_ratio": data.get('PEGRatio', 'N/A'),
                    "book_value": data.get('BookValue', 'N/A'),
                    "dividend_yield": data.get('DividendYield', 'N/A'),
                    "eps": data.get('EPS', 'N/A'),
                    "revenue_ttm": data.get('RevenueTTM', 'N/A'),
                    "profit_margin": data.get('ProfitMargin', 'N/A'),
                    "operating_margin": data.get('OperatingMarginTTM', 'N/A'),
                    "return_on_assets": data.get('ReturnOnAssetsTTM', 'N/A'),
                    "return_on_equity": data.get('ReturnOnEquityTTM', 'N/A'),
                    "beta": data.get('Beta', 'N/A'),
                    "52_week_high": data.get('52WeekHigh', 'N/A'),
                    "52_week_low": data.get('52WeekLow', 'N/A'),
                    "50_day_ma": data.get('50DayMovingAverage', 'N/A'),
                    "200_day_ma": data.get('200DayMovingAverage', 'N/A'),
                    "analyst_target_price": data.get('AnalystTargetPrice', 'N/A'),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched company overview for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching company overview: {str(e)}"}
    
    def fetch_technical_indicator(self, symbol: str, indicator: str = "RSI", 
                                  interval: str = "daily", time_period: int = 14) -> Dict[str, Any]:
        """
        Fetch technical indicators.
        indicator: RSI, MACD, SMA, EMA, STOCH, ADX, CCI, AROON, BBANDS, etc.
        interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching {indicator} for {symbol}...", total=None)
            
            try:
                params = {
                    'function': indicator.upper(),
                    'symbol': symbol,
                    'interval': interval,
                    'time_period': time_period,
                    'series_type': 'close'
                }
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                # Find the technical analysis key (varies by indicator)
                tech_key = None
                for key in data.keys():
                    if 'Technical Analysis' in key:
                        tech_key = key
                        break
                
                if not tech_key or not data.get(tech_key):
                    progress.update(task, description=f"[red]No {indicator} data found for {symbol}")
                    return {"error": f"No {indicator} data found for {symbol}"}
                
                tech_data = data[tech_key]
                
                # Get latest values
                latest_values = []
                for date, values in sorted(tech_data.items(), reverse=True)[:30]:
                    entry = {'date': date}
                    entry.update({k: float(v) for k, v in values.items()})
                    latest_values.append(entry)
                
                result = {
                    "symbol": symbol,
                    "indicator": indicator.upper(),
                    "interval": interval,
                    "time_period": time_period,
                    "latest_value": latest_values[0] if latest_values else {},
                    "values": latest_values,
                    "metadata": data.get('Meta Data', {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched {indicator} for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching technical indicator: {str(e)}"}
    
    def fetch_market_news_sentiment(self, tickers: str = None, topics: str = None, 
                                    time_from: str = None, time_to: str = None,
                                    limit: int = 50) -> Dict[str, Any]:
        """
        Fetch market news and sentiment data.
        tickers: comma-separated list of tickers (e.g., "AAPL,MSFT")
        topics: blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets, etc.
        time_from/time_to: YYYYMMDDTHHMM format
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Fetching market news and sentiment...", total=None)
            
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'limit': limit
                }
                
                if tickers:
                    params['tickers'] = tickers
                if topics:
                    params['topics'] = topics
                if time_from:
                    params['time_from'] = time_from
                if time_to:
                    params['time_to'] = time_to
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                feed = data.get('feed', [])
                
                if not feed:
                    progress.update(task, description="[yellow]No news articles found")
                    return {"articles": [], "message": "No news articles found"}
                
                articles = []
                for article in feed:
                    processed_article = {
                        "title": article.get('title', 'N/A'),
                        "url": article.get('url', 'N/A'),
                        "time_published": article.get('time_published', 'N/A'),
                        "authors": article.get('authors', []),
                        "summary": article.get('summary', 'N/A'),
                        "source": article.get('source', 'N/A'),
                        "category_within_source": article.get('category_within_source', 'N/A'),
                        "overall_sentiment_score": article.get('overall_sentiment_score', 0),
                        "overall_sentiment_label": article.get('overall_sentiment_label', 'neutral'),
                        "ticker_sentiment": article.get('ticker_sentiment', [])
                    }
                    articles.append(processed_article)
                
                # Calculate aggregate sentiment if tickers provided
                sentiment_summary = {}
                if tickers:
                    for ticker in tickers.split(','):
                        ticker = ticker.strip()
                        ticker_scores = []
                        for article in articles:
                            for ts in article.get('ticker_sentiment', []):
                                if ts.get('ticker') == ticker:
                                    ticker_scores.append(float(ts.get('ticker_sentiment_score', 0)))
                        
                        if ticker_scores:
                            sentiment_summary[ticker] = {
                                "average_sentiment": float(np.mean(ticker_scores)),
                                "sentiment_std": float(np.std(ticker_scores)),
                                "article_count": len(ticker_scores)
                            }
                
                result = {
                    "total_articles": len(articles),
                    "articles": articles,
                    "sentiment_summary": sentiment_summary,
                    "items": data.get('items', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched {len(articles)} news articles")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching news sentiment: {str(e)}"}
    
    def fetch_earnings(self, symbol: str) -> Dict[str, Any]:
        """Fetch earnings data for a stock."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Fetching earnings for {symbol}...", total=None)
            
            try:
                params = {
                    'function': 'EARNINGS',
                    'symbol': symbol
                }
                
                data = self._make_request(params)
                
                if "error" in data:
                    progress.update(task, description=f"[red]{data['error']}")
                    return data
                
                annual_earnings = data.get('annualEarnings', [])
                quarterly_earnings = data.get('quarterlyEarnings', [])
                
                result = {
                    "symbol": symbol,
                    "annual_earnings": annual_earnings[:5],  # Last 5 years
                    "quarterly_earnings": quarterly_earnings[:8],  # Last 8 quarters
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully fetched earnings for {symbol}")
                return result
                
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {"error": f"Error fetching earnings: {str(e)}"}


class WebToolsWrapper:
    """Web tools wrapper for web search and calculator functionality."""
    
    def __init__(self, api_key: str = None):
        """Initialize web tools wrapper with Tavily API key."""
        self.api_key = api_key or TAVILY_API_KEY
        if not self.api_key:
            console.print("[bold red]Warning: Tavily API key not found. Web search functionality will be limited.[/bold red]")
        else:
            console.print("[bold green]Web Tools Wrapper initialized[/bold green]")
    
    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Performs a web search using the Tavily API to get real-time information from the internet.
        
        Args:
            query (str): The search query to look up on the web.
            
        Returns:
            Dict[str, Any]: A dictionary containing search results and metadata.
            
        Use this tool when you need:
        - Current information or recent events
        - Real-time data (weather, news, stock prices, etc.)
        - Information not in your training data
        - To verify or fact-check current information
        
        Example:
            web_search("current Tesla stock news")
            web_search("latest AI trends 2025")
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Searching web for: {query}...", total=None)
            
            if not self.api_key:
                progress.update(task, description="[red]Tavily API key not configured")
                return {
                    "error": "TAVILY_API_KEY not found in environment variables. Please set it in your .env file.",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            
            try:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 3
                }
                
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract answer if available
                answer = data.get("answer", "")
                
                # Extract results
                results = []
                if "results" in data:
                    for result in data["results"][:3]:
                        results.append({
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0)
                        })
                
                result_data = {
                    "query": query,
                    "answer": answer,
                    "results": results,
                    "total_results": len(results),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Successfully found {len(results)} results")
                return result_data
                
            except requests.exceptions.RequestException as e:
                progress.update(task, description=f"[red]Request error: {str(e)}")
                return {
                    "error": f"Error performing web search: {str(e)}",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                progress.update(task, description=f"[red]Unexpected error: {str(e)}")
                return {
                    "error": f"Unexpected error: {str(e)}",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
    
    def calculator(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluates mathematical expressions and returns the result.
        
        Args:
            expression (str): A mathematical expression to evaluate (e.g., "2 + 2", "15 * 3.5").
            
        Returns:
            Dict[str, Any]: A dictionary containing the calculation result and metadata.
            
        Use this tool when you need to:
        - Perform arithmetic calculations
        - Add, subtract, multiply, or divide numbers
        - Calculate percentages or other mathematical operations
        - Compute financial ratios and metrics
        
        Supported operations: +, -, *, /, //, %, **
        
        Example:
            calculator("10 + 5")  # Returns {"result": "15", ...}
            calculator("100 / 4")  # Returns {"result": "25.0", ...}
            calculator("2 ** 8")   # Returns {"result": "256", ...}
        
        Note: For security, this only evaluates mathematical expressions and does not execute arbitrary code.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Calculating: {expression}...", total=None)
            
            try:
                # Clean the expression
                expression = expression.strip()
                
                # Only allow specific characters (numbers, operators, parentheses, decimal point, spaces)
                allowed_chars = set("0123456789+-*/().% ")
                if not all(c in allowed_chars for c in expression):
                    progress.update(task, description="[red]Invalid characters in expression")
                    return {
                        "error": "Expression contains invalid characters. Only use numbers and basic operators (+, -, *, /, %, ()).",
                        "expression": expression,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Evaluate the expression safely using sympify
                result = sympify(expression).evalf()
                
                # Convert the numeric result to a string for consistent return type
                result_str = str(result)
                
                result_data = {
                    "expression": expression,
                    "result": result_str,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                progress.update(task, description=f"[green]Result: {result_str}")
                return result_data
                
            except ZeroDivisionError:
                progress.update(task, description="[red]Division by zero error")
                return {
                    "error": "Division by zero.",
                    "expression": expression,
                    "timestamp": datetime.now().isoformat()
                }
            except SyntaxError:
                progress.update(task, description="[red]Syntax error")
                return {
                    "error": "Invalid mathematical expression syntax.",
                    "expression": expression,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                progress.update(task, description=f"[red]Error: {str(e)}")
                return {
                    "error": f"Error evaluating expression: {str(e)}",
                    "expression": expression,
                    "timestamp": datetime.now().isoformat()
                }

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

@tool
def search_web(query: str) -> Union[Dict, str]:
    """Performs a web search to get real-time information from the internet."""
    try:
        wrapper = WebToolsWrapper()
        result = wrapper.web_search(query)
        
        if "error" in result:
            return result["error"]
        
        return {
            'answer': result.get('answer', ''),
            'results': result.get('results', []),
            'total_results': result.get('total_results', 0)
        }
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool
def calculate(expression: str) -> Union[Dict, str]:
    """Safely evaluates mathematical expressions and returns the result."""
    try:
        wrapper = WebToolsWrapper()
        result = wrapper.calculator(expression)
        
        if "error" in result:
            return result["error"]
        
        return {
            'expression': result.get('expression', ''),
            'result': result.get('result', ''),
            'success': result.get('success', False)
        }
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

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
        self.tools = [get_stock_prices, get_financial_metrics, search_web, calculate]
        self.llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        self.data_wrapper = YahooFinanceDataWrapper()
        self.web_tools_wrapper = WebToolsWrapper()
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
        3. **search_web**: Performs web searches to get real-time information from the internet.
        4. **calculate**: Evaluates mathematical expressions for financial calculations.

        ### Your Task:
        1. Use the provided stock symbol to query the tools and collect data.
        2. Use search_web to get the latest news and market sentiment about the stock.
        3. Use calculate when you need to perform financial calculations (ratios, percentages, etc.).
        4. Analyze the results and identify trends and patterns.
        5. Provide a comprehensive summary with specific recommendations.

        ### Output Format:
        Respond with:
        "stock": "<Stock Symbol>",
        "price_analysis": "<Analysis of price trends and technical indicators>",
        "financial_analysis": "<Analysis of financial metrics and valuation>",
        "market_sentiment": "<Analysis based on web search results>",
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
            
        # Create recommendations directory if it doesn't exist
        os.makedirs('recommendations', exist_ok=True)
        
        # Generate markdown content
        markdown_content = f"""
# Investment Analysis for {stock_symbol}

## Analysis Summary
{analysis['ai_analysis']}

## Recommendations
"""
        for i, rec in enumerate(analysis['recommendations'], 1):
            markdown_content += f"{i}. {rec}\n"
            
        markdown_content += f"\n\n*Generated on: {datetime.now().isoformat()}*"
        
        # Save markdown file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recommendations/{stock_symbol}_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write(markdown_content)
        print(f"\nAnalysis saved to: {filename}")
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
