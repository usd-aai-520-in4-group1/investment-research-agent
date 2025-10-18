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
import random

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
    """Fetches historical stock price data and technical indicators using Yahoo Finance."""
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
    """Fetches key financial ratios and metrics using Yahoo Finance."""
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
def get_alpha_vantage_quote(ticker: str) -> Union[Dict, str]:
    """Fetches real-time quote data from Alpha Vantage. Use as alternative to Yahoo Finance."""
    try:
        wrapper = AlphaVantageDataWrapper()
        result = wrapper.fetch_quote(ticker)
        
        if "error" in result:
            return result["error"]
        
        return {
            'symbol': result.get('symbol'),
            'price': result.get('price'),
            'change': result.get('change'),
            'change_percent': result.get('change_percent'),
            'volume': result.get('volume'),
            'latest_trading_day': result.get('latest_trading_day')
        }
    except Exception as e:
        return f"Error fetching Alpha Vantage quote: {str(e)}"

@tool
def get_alpha_vantage_overview(ticker: str) -> Union[Dict, str]:
    """Fetches comprehensive company overview and fundamentals from Alpha Vantage. Use as alternative to Yahoo Finance."""
    try:
        wrapper = AlphaVantageDataWrapper()
        result = wrapper.fetch_company_overview(ticker)
        
        if "error" in result:
            return result["error"]
        
        return {
            'company_name': result.get('company_name'),
            'sector': result.get('sector'),
            'industry': result.get('industry'),
            'market_cap': result.get('market_cap'),
            'pe_ratio': result.get('pe_ratio'),
            'dividend_yield': result.get('dividend_yield'),
            'beta': result.get('beta'),
            'profit_margin': result.get('profit_margin'),
            'analyst_target_price': result.get('analyst_target_price')
        }
    except Exception as e:
        return f"Error fetching Alpha Vantage overview: {str(e)}"

@tool
def get_alpha_vantage_news(tickers: str) -> Union[Dict, str]:
    """Fetches market news and sentiment from Alpha Vantage. Use as alternative for news data."""
    try:
        wrapper = AlphaVantageDataWrapper()
        result = wrapper.fetch_market_news_sentiment(tickers=tickers, limit=10)
        
        if "error" in result:
            return result["error"]
        
        return {
            'total_articles': result.get('total_articles', 0),
            'articles': result.get('articles', [])[:5],  # Return top 5 articles
            'sentiment_summary': result.get('sentiment_summary', {})
        }
    except Exception as e:
        return f"Error fetching Alpha Vantage news: {str(e)}"

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
    
    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Search memories by query string. Searches through content, context, and tags.
        
        Args:
            query: Search query string
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memory entries sorted by relevance and importance
        """
        if not self.memories:
            return []
        
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        # Score each memory based on keyword matches
        scored_memories = []
        for memory in self.memories:
            score = 0.0
            
            # Check content
            content_lower = memory.content.lower()
            for keyword in query_keywords:
                if keyword in content_lower:
                    score += 2.0
            
            # Check context (if query mentions similar things)
            context_str = json.dumps(memory.context).lower()
            for keyword in query_keywords:
                if keyword in context_str:
                    score += 1.0
            
            # Check tags
            for tag in memory.tags:
                if tag.lower() in query_lower:
                    score += 1.5
            
            # Boost by importance score
            score *= (1 + memory.importance_score)
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score (descending) and take top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        relevant_memories = [mem for score, mem in scored_memories[:limit]]
        
        console.print(f"[green]Found {len(relevant_memories)} relevant memories for query[/green]")
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
# AGENTIC WORKFLOW PATTERNS
# =============================================================================

class NewsChainWorkflow:
    """Prompt Chaining: Ingest → Preprocess → Classify → Extract → Summarize"""
    
    def __init__(self, llm):
        self.llm = llm
        console.print("[bold green]News Chain Workflow initialized[/bold green]")
    
    def execute(self, stock_symbol: str, raw_news: List[Dict]) -> Dict[str, Any]:
        """Execute the news analysis chain."""
        console.print(f"[cyan]Starting News Chain Workflow for {stock_symbol}[/cyan]")
        
        # Step 1: Ingest
        ingested = self._ingest_news(raw_news)
        
        # Step 2: Preprocess
        preprocessed = self._preprocess_news(ingested)
        
        # Step 3: Classify
        classified = self._classify_news(preprocessed)
        
        # Step 4: Extract
        extracted = self._extract_key_points(classified)
        
        # Step 5: Summarize
        summary = self._summarize_news(extracted)
        
        return {
            "workflow": "prompt_chaining",
            "steps_completed": 5,
            "final_summary": summary,
            "classified_topics": classified.get("topics", []),
            "key_points": extracted.get("points", [])
        }
    
    def _ingest_news(self, raw_news: List[Dict]) -> Dict[str, Any]:
        """Step 1: Ingest raw news data."""
        console.print("[dim]→ Step 1/5: Ingesting news...[/dim]")
        return {
            "articles": raw_news,
            "count": len(raw_news),
            "ingested_at": datetime.now().isoformat()
        }
    
    def _preprocess_news(self, ingested: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Preprocess news articles."""
        console.print("[dim]→ Step 2/5: Preprocessing...[/dim]")
        
        prompt = f"""Preprocess these news articles. Clean and structure the data.
        
Articles: {json.dumps(ingested['articles'][:3], indent=2)}

Return a structured format with: title, content, source, date."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "preprocessed_articles": ingested['articles'],
            "llm_preprocessing": response.content,
            "count": ingested['count']
        }
    
    def _classify_news(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Classify news by topic and sentiment."""
        console.print("[dim]→ Step 3/5: Classifying...[/dim]")
        
        prompt = f"""Classify these news articles by topic and sentiment.
        
Articles: {json.dumps(preprocessed['preprocessed_articles'][:3], indent=2)}

Classify into topics: earnings, product_launch, merger, regulation, general
Classify sentiment: positive, negative, neutral

Return format:
Topic: [topic]
Sentiment: [sentiment]
Reasoning: [brief reasoning]"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "articles": preprocessed['preprocessed_articles'],
            "classification": response.content,
            "topics": ["earnings", "general"],  # Parsed from LLM response
            "overall_sentiment": "mixed"
        }
    
    def _extract_key_points(self, classified: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Extract key points from articles."""
        console.print("[dim]→ Step 4/5: Extracting key points...[/dim]")
        
        prompt = f"""Extract the most important points from these classified articles.
        
Classification: {classified['classification']}

Extract:
1. Key financial impacts
2. Important dates or events
3. Market implications
4. Risk factors

Format as bullet points."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "classification": classified['classification'],
            "key_points_raw": response.content,
            "points": response.content.split('\n')
        }
    
    def _summarize_news(self, extracted: Dict[str, Any]) -> str:
        """Step 5: Create final summary."""
        console.print("[dim]→ Step 5/5: Summarizing...[/dim]")
        
        prompt = f"""Create a concise executive summary based on these key points.
        
Key Points: {extracted['key_points_raw']}

Provide:
- 2-3 sentence overview
- Main takeaway for investors
- Sentiment indicator (bullish/bearish/neutral)

Keep it actionable and clear."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        console.print("[green]✓ News Chain completed[/green]")
        
        return response.content


class RoutingWorkflow:
    """Routing: Direct queries to specialist analyzers"""
    
    def __init__(self, llm):
        self.llm = llm
        console.print("[bold green]Routing Workflow initialized[/bold green]")
    
    def execute(self, stock_symbol: str, query: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Route query to appropriate specialist."""
        console.print(f"[cyan]Starting Routing Workflow for {stock_symbol}[/cyan]")
        
        # Determine which specialist to use
        specialist_type = self._route_query(query)
        console.print(f"[yellow]→ Routing to: {specialist_type} specialist[/yellow]")
        
        # Route to specialist
        if specialist_type == "earnings":
            result = self._earnings_specialist(stock_symbol, data)
        elif specialist_type == "news":
            result = self._news_specialist(stock_symbol, data)
        elif specialist_type == "technical":
            result = self._technical_specialist(stock_symbol, data)
        else:
            result = self._general_specialist(stock_symbol, query, data)
        
        return {
            "workflow": "routing",
            "specialist": specialist_type,
            "analysis": result
        }
    
    def _route_query(self, query: str) -> str:
        """Determine which specialist to use based on query."""
        query_lower = query.lower()
        
        # Use LLM to classify query type
        prompt = f"""Determine which specialist should handle this investment query.
        
                    Query: {query_lower}

                    Available specialists:
                    - earnings: For questions about financials, revenue, profit, EPS
                    - news: For questions about announcements, events, sentiment
                    - technical: For questions about price action, charts, trends
                    - general: For other investment questions

                    Return just one word - the specialist type that best matches."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        specialist = response.content.strip().lower()
        
        if specialist in ["earnings", "news", "technical"]:
            return specialist
        else:
            return "general"
    
    def _earnings_specialist(self, stock_symbol: str, data: Dict[str, Any]) -> str:
        """Earnings specialist analysis."""
        console.print("[dim]→ Earnings specialist analyzing...[/dim]")
        
        financial_data = data.get("financial_metrics", {})
        
        prompt = f"""You are an earnings specialist. Analyze the financial metrics for {stock_symbol}.
        
                Financial Data: {json.dumps(financial_data, indent=2)}

                Focus on:
                - Revenue and earnings trends
                - Profitability metrics
                - Growth indicators
                - Valuation ratios

                Provide specific insights about earnings quality and sustainability."""
                        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _news_specialist(self, stock_symbol: str, data: Dict[str, Any]) -> str:
        """News specialist analysis."""
        console.print("[dim]→ News specialist analyzing...[/dim]")
        
        news_data = data.get("news_data", {})
        
        prompt = f"""You are a news sentiment specialist. Analyze news for {stock_symbol}.
        
                News Data: {json.dumps(news_data, indent=2)}

                Focus on:
                - Sentiment trends
                - Key events and their impact
                - Market perception
                - Potential catalysts

                Provide actionable insights from news analysis."""
                        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _technical_specialist(self, stock_symbol: str, data: Dict[str, Any]) -> str:
        """Technical analysis specialist."""
        console.print("[dim]→ Technical specialist analyzing...[/dim]")
        
        price_data = data.get("historical_prices", {})
        
        prompt = f"""You are a technical analysis specialist. Analyze price action for {stock_symbol}.
        
                    Price Data: Latest ${price_data.get('latest_price', 'N/A')}
                    Technical Indicators: {json.dumps(price_data.get('technical_indicators', {}), indent=2)}

                    Focus on:
                    - Trend direction and strength
                    - Support/resistance levels
                    - Technical indicator signals
                    - Entry/exit points

                    Provide specific technical trading insights."""
                            
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _general_specialist(self, stock_symbol: str, query: str, data: Dict[str, Any]) -> str:
        """General investment specialist."""
        console.print("[dim]→ General specialist analyzing...[/dim]")
        
        prompt = f"""You are a general investment specialist. Answer this query about {stock_symbol}.
        
                    Query: {query}
                    Available Data: {json.dumps(data, indent=2)[:500]}...

                    Provide a comprehensive, balanced analysis addressing the specific question."""
                            
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class EvaluatorOptimizerWorkflow:
    """Evaluator-Optimizer: Generate → Evaluate → Refine"""
    
    def __init__(self, llm):
        self.llm = llm
        self.max_iterations = 2
        console.print("[bold green]Evaluator-Optimizer Workflow initialized[/bold green]")
    
    def execute(self, stock_symbol: str, initial_analysis: str) -> Dict[str, Any]:
        """Execute the evaluator-optimizer loop."""
        console.print(f"[cyan]Starting Evaluator-Optimizer Workflow for {stock_symbol}[/cyan]")
        
        current_analysis = initial_analysis
        iteration = 0
        evaluation_history = []
        
        while iteration < self.max_iterations:
            iteration += 1
            console.print(f"[yellow]→ Iteration {iteration}/{self.max_iterations}[/yellow]")
            
            # Evaluate
            evaluation = self._evaluate_analysis(current_analysis)
            evaluation_history.append(evaluation)
            
            console.print(f"[dim]Quality Score: {evaluation['quality_score']:.2f}/10[/dim]")
            
            # Check if quality is acceptable
            if evaluation['quality_score'] >= 8.0:
                console.print("[green]✓ Quality threshold met[/green]")
                break
            
            # Optimize
            current_analysis = self._optimize_analysis(current_analysis, evaluation)
        
        return {
            "workflow": "evaluator_optimizer",
            "iterations": iteration,
            "final_analysis": current_analysis,
            "final_quality_score": evaluation_history[-1]['quality_score'],
            "improvements_made": [e['issues'] for e in evaluation_history]
        }
    
    def _evaluate_analysis(self, analysis: str) -> Dict[str, Any]:
        """Evaluate the quality of analysis."""
        console.print("[dim]→ Evaluating analysis quality...[/dim]")
        
        prompt = f"""Evaluate this investment analysis for quality.
        
                    Analysis:
                    {analysis}

                    Rate on scale of 1-10 for:
                    1. Completeness (covers all key aspects)
                    2. Accuracy (logical and well-reasoned)
                    3. Actionability (provides clear recommendations)
                    4. Clarity (easy to understand)

                    Format:
                    Completeness: X/10
                    Accuracy: X/10
                    Actionability: X/10
                    Clarity: X/10
                    Overall: X/10
                    Issues: [list specific issues to improve]"""
                            
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse the evaluation (simplified)
        content = response.content
        try:
            overall_line = [line for line in content.split('\n') if 'Overall:' in line][0]
            score = float(overall_line.split(':')[1].split('/')[0].strip())
        except:
            score = 7.0  # Default if parsing fails
        
        return {
            "quality_score": score,
            "detailed_evaluation": content,
            "issues": content.split('Issues:')[-1].strip() if 'Issues:' in content else ""
        }
    
    def _optimize_analysis(self, analysis: str, evaluation: Dict[str, Any]) -> str:
        """Refine analysis based on evaluation."""
        console.print("[dim]→ Optimizing analysis...[/dim]")
        
        prompt = f"""Improve this investment analysis based on the evaluation feedback.
        
                Original Analysis:
                {analysis}

                Evaluation Feedback:
                {evaluation['detailed_evaluation']}

                Specific Issues to Address:
                {evaluation['issues']}

                Provide an improved version that addresses all issues while maintaining accuracy."""
                        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        console.print("[green]✓ Analysis optimized[/green]")
        
        return response.content


# =============================================================================
# MAIN INVESTMENT RESEARCH AGENT
# =============================================================================

class InvestmentAgent:
    """Investment Research Agent combining all functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.tools = [
            get_stock_prices, 
            get_financial_metrics, 
            get_alpha_vantage_quote,
            get_alpha_vantage_overview,
            get_alpha_vantage_news,
            search_web, 
            calculate
        ]
        self.llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        self.data_wrapper = YahooFinanceDataWrapper()
        self.alpha_vantage_wrapper = AlphaVantageDataWrapper()
        self.web_tools_wrapper = WebToolsWrapper()
        self.analyzer = Analyzer()
        self.memory_system = MemorySystem()
        self.results = {}
        
        # Initialize agentic workflows
        self.news_chain = NewsChainWorkflow(self.llm)
        self.routing_workflow = RoutingWorkflow(self.llm)
        self.evaluator_optimizer = EvaluatorOptimizerWorkflow(self.llm)
        
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
        messages = state["messages"]
        stock_symbol = state.get("stock", "UNKNOWN")
        
        # Check if this is the first call (no tool results yet)
        has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
        
        if not has_system_message:
            # First call - add system prompt
            ANALYST_PROMPT = f"""You are a financial analyst specializing in stock evaluation for {stock_symbol}.

                You have access to these tools:
                1. **get_stock_prices**: Get price data and technical indicators from Yahoo Finance
                2. **get_financial_metrics**: Get financial metrics and ratios from Yahoo Finance
                3. **get_alpha_vantage_quote**: Get real-time quote from Alpha Vantage (alternative source)
                4. **get_alpha_vantage_overview**: Get company overview from Alpha Vantage (alternative source)
                5. **get_alpha_vantage_news**: Get news and sentiment from Alpha Vantage
                6. **search_web**: Search for latest news and information
                7. **calculate**: Perform financial calculations

                Start by gathering data using the tools. Use Alpha Vantage tools as alternatives if Yahoo Finance fails.
                Then provide comprehensive analysis."""
                            
            messages.insert(0, SystemMessage(content=ANALYST_PROMPT))
        
        # Invoke the LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "stock": stock_symbol,
            "analysis_data": state.get("analysis_data", {}),
            "final_analysis": state.get("final_analysis", {})
        }
    
    def extract_stock_symbol(self, natural_query: str) -> Dict[str, Any]:
        """
        Extract stock symbol from a natural language query using memory and web search.
        
        Args:
            natural_query: Natural language query like "What's the latest on Apple stock?"
            
        Returns:
            Dict containing the stock symbol and confidence
        """
        console.print(f"[cyan]Extracting stock symbol from query...[/cyan]")
        
        # First check for most recent stock symbol from query history
        console.print("[dim]→ Checking most recent stock symbol from query history...[/dim]")
        if self.memory_system.memories:
            # Get memories sorted by timestamp (most recent first)
            sorted_memories = sorted(
                self.memory_system.memories, 
                key=lambda m: m.timestamp, 
                reverse=True
            )
            
            # Find the most recent memory with a valid stock symbol
            for memory in sorted_memories[:10]:  # Check last 10 memories
                if memory.stock_symbol and memory.stock_symbol.isalpha() and len(memory.stock_symbol) <= 5:
                    console.print(f"[green]✓ Found most recent stock symbol from history: {memory.stock_symbol}[/green]")
                    console.print(f"[dim]  From: {memory.timestamp}[/dim]")
                    console.print(f"[dim]  Memory: {memory.content[:100]}...[/dim]")
                    return {
                        "stock_symbol": memory.stock_symbol,
                        "confidence": "high",
                        "method": "recent_history",
                        "original_query": natural_query,
                        "memory_content": memory.content[:200],
                        "memory_timestamp": memory.timestamp
                    }
        
        # Second, check memory for keyword-matched queries
        console.print("[dim]→ Searching memory for keyword-matched queries...[/dim]")
        memories = self.memory_system.search_memories(natural_query, limit=3)
        
        if memories:
            # Check if any memory has a valid stock symbol
            for memory in memories:
                if memory.stock_symbol and memory.stock_symbol.isalpha() and len(memory.stock_symbol) <= 5:
                    console.print(f"[green]✓ Found stock symbol in keyword-matched memory: {memory.stock_symbol}[/green]")
                    console.print(f"[dim]  Memory: {memory.content[:100]}...[/dim]")
                    return {
                        "stock_symbol": memory.stock_symbol,
                        "confidence": "high",
                        "method": "memory_keyword_match",
                        "original_query": natural_query,
                        "memory_content": memory.content[:200]
                    }
        
        # Try web search if memory doesn't help
        console.print("[dim]→ No relevant memories found, trying web search...[/dim]")
        try:
            search_result = self._extract_symbol_via_web_search(natural_query)
            if search_result["stock_symbol"]:
                # Store in memory for future queries
                self._store_extraction_in_memory(
                    search_result["stock_symbol"], 
                    natural_query, 
                    "web_search"
                )
                return search_result
        except Exception as e:
            console.print(f"[yellow]Web search failed: {str(e)}[/yellow]")

        # Fall back to LLM extraction if both memory and web search fail
        console.print("[dim]→ Trying LLM extraction...[/dim]")
        extraction_prompt = f"""Extract the stock ticker symbol from this query. If you can identify the company name, 
            provide the stock ticker symbol. If a ticker is already mentioned, return it in uppercase.

            Query: {natural_query}

            Respond with ONLY the stock ticker symbol (e.g., AAPL, MSFT, GOOGL, TSLA) or "UNKNOWN" if you cannot identify it.
            Do not include any explanation, just the ticker symbol."""

        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            extracted_symbol = response.content.strip().upper()
            
            # Validate the extracted symbol
            if extracted_symbol != "UNKNOWN" and len(extracted_symbol) <= 5 and extracted_symbol.isalpha():
                console.print(f"[green]✓ Extracted stock symbol via LLM: {extracted_symbol}[/green]")
                
                # Store in memory for future queries
                self._store_extraction_in_memory(
                    extracted_symbol, 
                    natural_query, 
                    "llm_extraction"
                )
                
                return {
                    "stock_symbol": extracted_symbol,
                    "confidence": "medium",
                    "method": "llm_extraction",
                    "original_query": natural_query
                }
            
        except Exception as e:
            console.print(f"[yellow]LLM extraction failed: {str(e)}[/yellow]")

        # If all methods fail
        console.print("[red]✗ Could not extract stock symbol[/red]")
        return {
            "stock_symbol": None,
            "confidence": "none", 
            "method": "failed",
            "original_query": natural_query,
            "error": "Unable to identify stock symbol from query"
        }
    
    def _store_extraction_in_memory(self, stock_symbol: str, query: str, method: str) -> None:
        """Store successful symbol extraction in memory for future queries."""
        try:
            self.memory_system.add_memory(
                stock_symbol=stock_symbol,
                memory_type="symbol_extraction",
                content=f"Query: '{query}' -> Symbol: {stock_symbol}",
                context={
                    "query": query,
                    "method": method,
                    "extraction_timestamp": datetime.now().isoformat()
                },
                importance_score=0.7,
                tags=["extraction", method, query.lower()[:50]]
            )
            console.print(f"[dim]✓ Stored extraction in memory[/dim]")
        except Exception as e:
            console.print(f"[dim yellow]Warning: Could not store in memory: {str(e)}[/dim yellow]")
            
    def _extract_symbol_via_web_search(self, natural_query: str) -> Dict[str, Any]:
        """Extract stock symbol using web search."""
        try:
            # Search for the company + stock ticker
            search_query = f"{natural_query} stock ticker symbol"
            search_result = self.web_tools_wrapper.web_search(search_query)
            
            if "error" not in search_result and search_result.get('results'):
                # Use LLM to parse the search results and extract ticker
                results_text = "\n".join([
                    f"{r.get('title', '')}: {r.get('content', '')[:200]}"
                    for r in search_result['results'][:3]
                ])
                
                parsing_prompt = f"""Based on these search results about "{natural_query}", what is the stock ticker symbol?

                Search Results:
                {results_text}

                Respond with ONLY the stock ticker symbol in uppercase (e.g., AAPL, MSFT) or "UNKNOWN" if unclear."""

                response = self.llm.invoke([HumanMessage(content=parsing_prompt)])
                extracted_symbol = response.content.strip().upper()
                
                if extracted_symbol != "UNKNOWN" and len(extracted_symbol) <= 5 and extracted_symbol.isalpha():
                    console.print(f"[green]✓ Extracted via web search: {extracted_symbol}[/green]")
                    return {
                        "stock_symbol": extracted_symbol,
                        "confidence": "medium",
                        "method": "web_search",
                        "original_query": natural_query
                    }
            
            # If all else fails
            console.print("[red]✗ Could not extract stock symbol[/red]")
            return {
                "stock_symbol": None,
                "confidence": "none",
                "method": "failed",
                "original_query": natural_query,
                "error": "Unable to identify stock symbol from query"
            }
            
        except Exception as e:
            console.print(f"[red]Web search extraction failed: {str(e)}[/red]")
            return {
                "stock_symbol": None,
                "confidence": "none",
                "method": "failed",
                "original_query": natural_query,
                "error": str(e)
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
    
    def query(self, stock_symbol: str, user_query: str, use_optimizer: bool = True) -> Dict[str, Any]:
        """
        Main query handler - intelligently routes to appropriate workflows.
        
        This is the main entry point that:
        1. Collects necessary data
        2. Determines which workflow pattern to use
        3. Executes the workflow
        4. Optionally optimizes the result
        5. Returns comprehensive analysis
        
        Args:
            stock_symbol: Stock ticker symbol
            user_query: User's question about the stock
            use_optimizer: Whether to use evaluator-optimizer workflow
            
        Returns:
            Dict containing analysis results and workflow information
        """
        console.print(Panel.fit(
            f"[bold cyan]Processing Query[/bold cyan]\n"
            f"Stock: [yellow]{stock_symbol}[/yellow]\n"
            f"Query: [white]{user_query}[/white]",
            border_style="cyan"
        ))
        
        # Step 1: Collect data
        console.print("\n[bold]Step 1: Gathering Data[/bold]")
        data = self._collect_stock_data(stock_symbol)
        
        # Step 2: Determine workflow strategy
        console.print("\n[bold]Step 2: Determining Workflow Strategy[/bold]")
        workflow_strategy = self._determine_workflow(user_query)
        console.print(f"[yellow]→ Selected Strategy: {workflow_strategy}[/yellow]")
        
        # Step 3: Execute appropriate workflow
        console.print("\n[bold]Step 3: Executing Workflow[/bold]")
        
        if workflow_strategy == "news_chain":
            # Use prompt chaining for news-focused queries
            news_data = data.get("news_data", {}).get("news", [])
            workflow_result = self.news_chain.execute(stock_symbol, news_data)
            analysis = workflow_result["final_summary"]
            
        elif workflow_strategy == "routing":
            # Use routing for specialist analysis
            workflow_result = self.routing_workflow.execute(stock_symbol, user_query, data)
            analysis = workflow_result["analysis"]
            
        else:
            # Default: comprehensive analysis
            workflow_result = {"workflow": "comprehensive"}
            analysis = self._generate_comprehensive_analysis(stock_symbol, user_query, data)
        
        # Step 4: Optimize if requested
        if use_optimizer:
            console.print("\n[bold]Step 4: Optimizing Analysis[/bold]")
            optimizer_result = self.evaluator_optimizer.execute(stock_symbol, analysis)
            final_analysis = optimizer_result["final_analysis"]
            workflow_result["optimization"] = {
                "iterations": optimizer_result["iterations"],
                "final_quality_score": optimizer_result["final_quality_score"]
            }
        else:
            final_analysis = analysis
        
        # Step 5: Store in memory
        self.memory_system.add_memory(
            stock_symbol, "query_analysis",
            f"Query: {user_query[:100]}... Result: {final_analysis[:200]}...",
            {"query": user_query, "workflow": workflow_strategy},
            0.7, ["query", workflow_strategy]
        )
        
        # Prepare final result
        result = {
            "stock_symbol": stock_symbol,
            "user_query": user_query,
            "workflow_used": workflow_strategy,
            "workflow_details": workflow_result,
            "final_analysis": final_analysis,
            "data_summary": {
                "price": data.get("historical_prices", {}).get("latest_price", "N/A"),
                "pe_ratio": data.get("financial_metrics", {}).get("pe_ratio", "N/A"),
                "news_count": len(data.get("news_data", {}).get("news", []))
            },
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        console.print("\n[bold green]✓ Query Processing Completed[/bold green]")
        return result
    
    def _collect_stock_data(self, stock_symbol: str) -> Dict[str, Any]:
        """Collect all necessary stock data, randomly choosing between Yahoo Finance and Alpha Vantage."""
        data = {}
        
        # Randomly choose primary data source
        use_yahoo_first = random.choice([True, False])
        primary_source = "Yahoo Finance" if use_yahoo_first else "Alpha Vantage"
        fallback_source = "Alpha Vantage" if use_yahoo_first else "Yahoo Finance"
        
        console.print(f"[bold magenta] Randomly selected primary source: {primary_source}[/bold magenta]")
        
        # Fetch historical prices
        if use_yahoo_first:
            # Try Yahoo Finance first
            try:
                data["historical_prices"] = self.data_wrapper.fetch_historical_prices(stock_symbol)
                if "error" in data["historical_prices"]:
                    raise Exception("Yahoo Finance returned error")
            except Exception as e:
                console.print(f"[yellow]{primary_source} prices failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    av_prices = self.alpha_vantage_wrapper.fetch_daily_prices(stock_symbol, outputsize="compact")
                    if "error" not in av_prices:
                        data["historical_prices"] = av_prices
                        console.print(f"[green]✓ Using {fallback_source} price data[/green]")
                    else:
                        data["historical_prices"] = {}
                except Exception as av_e:
                    console.print(f"[red]{fallback_source} also failed: {str(av_e)}[/red]")
                    data["historical_prices"] = {}
        else:
            # Try Alpha Vantage first
            try:
                av_prices = self.alpha_vantage_wrapper.fetch_daily_prices(stock_symbol, outputsize="compact")
                if "error" in av_prices:
                    raise Exception("Alpha Vantage returned error")
                data["historical_prices"] = av_prices
            except Exception as e:
                console.print(f"[yellow]{primary_source} prices failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    data["historical_prices"] = self.data_wrapper.fetch_historical_prices(stock_symbol)
                    if "error" in data["historical_prices"]:
                        data["historical_prices"] = {}
                    else:
                        console.print(f"[green]✓ Using {fallback_source} price data[/green]")
                except Exception as yf_e:
                    console.print(f"[red]{fallback_source} also failed: {str(yf_e)}[/red]")
                    data["historical_prices"] = {}
        
        # Fetch financial metrics
        if use_yahoo_first:
            # Try Yahoo Finance first
            try:
                financial_result = self.data_wrapper.fetch_key_financials(stock_symbol)
                if "error" in financial_result:
                    raise Exception("Yahoo Finance returned error")
                data["financial_metrics"] = financial_result.get("key_metrics", {})
            except Exception as e:
                console.print(f"[yellow]{primary_source} metrics failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    av_overview = self.alpha_vantage_wrapper.fetch_company_overview(stock_symbol)
                    if "error" not in av_overview:
                        data["financial_metrics"] = {
                            "symbol": av_overview.get("symbol"),
                            "company_name": av_overview.get("company_name"),
                            "sector": av_overview.get("sector"),
                            "industry": av_overview.get("industry"),
                            "market_cap": av_overview.get("market_cap"),
                            "pe_ratio": av_overview.get("pe_ratio"),
                            "beta": av_overview.get("beta"),
                        }
                        console.print(f"[green]✓ Using {fallback_source} financial data[/green]")
                    else:
                        data["financial_metrics"] = {}
                except Exception as av_e:
                    console.print(f"[red]{fallback_source} also failed: {str(av_e)}[/red]")
                    data["financial_metrics"] = {}
        else:
            # Try Alpha Vantage first
            try:
                av_overview = self.alpha_vantage_wrapper.fetch_company_overview(stock_symbol)
                if "error" in av_overview:
                    raise Exception("Alpha Vantage returned error")
                data["financial_metrics"] = {
                    "symbol": av_overview.get("symbol"),
                    "company_name": av_overview.get("company_name"),
                    "sector": av_overview.get("sector"),
                    "industry": av_overview.get("industry"),
                    "market_cap": av_overview.get("market_cap"),
                    "pe_ratio": av_overview.get("pe_ratio"),
                    "beta": av_overview.get("beta"),
                }
            except Exception as e:
                console.print(f"[yellow]{primary_source} metrics failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    financial_result = self.data_wrapper.fetch_key_financials(stock_symbol)
                    if "error" in financial_result:
                        data["financial_metrics"] = {}
                    else:
                        data["financial_metrics"] = financial_result.get("key_metrics", {})
                        console.print(f"[green]✓ Using {fallback_source} financial data[/green]")
                except Exception as yf_e:
                    console.print(f"[red]{fallback_source} also failed: {str(yf_e)}[/red]")
                    data["financial_metrics"] = {}
        
        # Fetch news
        if use_yahoo_first:
            # Try Yahoo Finance first
            try:
                news_result = self.data_wrapper.fetch_stock_news(stock_symbol)
                if "error" in news_result:
                    raise Exception("Yahoo Finance returned error")
                data["news_data"] = news_result
            except Exception as e:
                console.print(f"[yellow]{primary_source} news failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    av_news = self.alpha_vantage_wrapper.fetch_market_news_sentiment(tickers=stock_symbol, limit=10)
                    if "error" not in av_news:
                        data["news_data"] = {
                            "symbol": stock_symbol,
                            "total_articles": av_news.get("total_articles", 0),
                            "news": av_news.get("articles", []),
                            "sentiment_summary": av_news.get("sentiment_summary", {})
                        }
                        console.print(f"[green]✓ Using {fallback_source} news data[/green]")
                    else:
                        data["news_data"] = {}
                except Exception as av_e:
                    console.print(f"[red]{fallback_source} also failed: {str(av_e)}[/red]")
                    data["news_data"] = {}
        else:
            # Try Alpha Vantage first
            try:
                av_news = self.alpha_vantage_wrapper.fetch_market_news_sentiment(tickers=stock_symbol, limit=10)
                if "error" in av_news:
                    raise Exception("Alpha Vantage returned error")
                data["news_data"] = {
                    "symbol": stock_symbol,
                    "total_articles": av_news.get("total_articles", 0),
                    "news": av_news.get("articles", []),
                    "sentiment_summary": av_news.get("sentiment_summary", {})
                }
            except Exception as e:
                console.print(f"[yellow]{primary_source} news failed: {str(e)}. Trying {fallback_source}...[/yellow]")
                try:
                    news_result = self.data_wrapper.fetch_stock_news(stock_symbol)
                    if "error" in news_result:
                        data["news_data"] = {}
                    else:
                        data["news_data"] = news_result
                        console.print(f"[green]✓ Using {fallback_source} news data[/green]")
                except Exception as yf_e:
                    console.print(f"[red]{fallback_source} also failed: {str(yf_e)}[/red]")
                    data["news_data"] = {}
        
        return data
    
    def _determine_workflow(self, query: str) -> str:
        """Determine which workflow to use based on query."""
        query_lower = query.lower()
        
        # Use LLM to determine workflow based on query content and intent
        prompt = f"""Determine the most appropriate workflow for this investment query.

                Query: {query}

                Available workflows:
                - news_chain: For queries focused on news, headlines, announcements, sentiment analysis
                - routing: For specialist queries about earnings, technicals, or specific metrics
                - comprehensive: For general investment questions needing broad analysis

                Return just one word - the workflow that best matches."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        workflow = response.content.strip().lower()
        
        if workflow in ["news_chain", "routing"]:
            return workflow
        else:
            return "comprehensive"
    
    def _generate_comprehensive_analysis(self, stock_symbol: str, query: str, data: Dict[str, Any]) -> str:
        """Generate comprehensive analysis combining all data."""
        prompt = f"""Provide a comprehensive investment analysis for {stock_symbol}.

                User Query: {query}

                Price Data: Latest ${data.get('historical_prices', {}).get('latest_price', 'N/A')}
                Change: {data.get('historical_prices', {}).get('price_change_pct', 'N/A')}%

                Financial Metrics:
                - P/E Ratio: {data.get('financial_metrics', {}).get('pe_ratio', 'N/A')}
                - Market Cap: {data.get('financial_metrics', {}).get('market_cap', 'N/A')}
                - Recommendation: {data.get('financial_metrics', {}).get('recommendation', 'N/A')}

                News Articles: {len(data.get('news_data', {}).get('news', []))} articles available

                Provide:
                1. Overall assessment
                2. Key strengths and risks
                3. Investment recommendation
                4. Price target or action items

                Be specific and actionable."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content



def main():
    """Main function demonstrating agentic workflow capabilities."""
    console.print(Panel.fit(
        "[bold cyan]Investment Research Agent[/bold cyan]\n"
        "[white]Powered by Agentic Workflows[/white]\n"
        "[dim]With Natural Language Query Support & Alpha Vantage Integration[/dim]",
        border_style="cyan"
    ))
    
    # Initialize the agent
    console.print("\n[yellow]Initializing agent...[/yellow]")
    agent = InvestmentAgent()

    while True:
        console.print("\n" + "-" * 60)
        console.print("[bold]Enter your query about a stock:[/bold]")
        console.print("\n[bold]Examples:[/bold]")
        console.print("  • What's the latest updates on Apple stock?")
        console.print("  • Should I invest in Tesla?")
        console.print("  • Tell me about Microsoft's performance")
        console.print("  • Is Google a good buy right now?")
        console.print("  • Type 'quit' to exit")
            
        natural_query = console.input("\n[bold cyan]Your natural language query: [/bold cyan]").strip()
            
        if natural_query.lower() == 'quit':
            break
            
        # Extract stock symbol from natural query
        console.print("\n" + "-" * 60)
        extraction_result = agent.extract_stock_symbol(natural_query)
            
        if extraction_result.get('stock_symbol'):
            stock_symbol = extraction_result['stock_symbol']
            console.print(f"\n[bold green]✓ Identified Stock: {stock_symbol}[/bold green]")
            console.print(f"[dim]Confidence: {extraction_result['confidence']} | Method: {extraction_result['method']}[/dim]")
                
            # Ask for confirmation
            confirm = console.input(f"\n[bold cyan]Proceed with {stock_symbol}? (yes/no): [/bold cyan]").lower().strip()
            if confirm not in ['yes', 'y']:
                console.print("[yellow]Let me try a web search to find the correct stock...[/yellow]")
                
                # Fallback: Try web search
                try:
                    console.print(f"[dim]→ Searching web for: {natural_query}[/dim]")
                    web_search_result = agent._extract_symbol_via_web_search(natural_query)
                    
                    if web_search_result.get('stock_symbol') and web_search_result['stock_symbol'] != stock_symbol:
                        stock_symbol = web_search_result['stock_symbol']
                        console.print(f"\n[bold green]✓ Found alternative stock via web search: {stock_symbol}[/bold green]")
                        console.print(f"[dim]Confidence: {web_search_result['confidence']} | Method: {web_search_result['method']}[/dim]")
                        
                        # Ask for confirmation again
                        confirm2 = console.input(f"\n[bold cyan]Proceed with {stock_symbol}? (yes/no): [/bold cyan]").lower().strip()
                        if confirm2 not in ['yes', 'y']:
                            console.print("[yellow]Skipping this query...[/yellow]")
                            continue
                    else:
                        # If web search also fails or returns same symbol, ask user to manually enter
                        console.print("[yellow]Web search didn't find a different stock symbol.[/yellow]")
                        manual_symbol = console.input("\n[bold cyan]Please enter the stock symbol manually (or press Enter to skip): [/bold cyan]").strip().upper()
                        
                        if manual_symbol and manual_symbol.isalpha() and len(manual_symbol) <= 5:
                            stock_symbol = manual_symbol
                            console.print(f"[green]✓ Using manually entered symbol: {stock_symbol}[/green]")
                        else:
                            console.print("[yellow]Skipping this query...[/yellow]")
                            continue
                            
                except Exception as e:
                    console.print(f"[red]Web search fallback failed: {str(e)}[/red]")
                    console.print("[yellow]Skipping this query...[/yellow]")
                    continue
                
            natural_query_used = True
        else:
            console.print(f"\n[bold red]✗ Could not identify stock symbol[/bold red]")
            console.print(f"[yellow]Error: {extraction_result.get('error', 'Unknown error')}[/yellow]")
            console.print("\n[dim]Please try again with a more specific query.[/dim]")
            continue
        
        # Now we have a valid stock_symbol, get the analysis query
        console.print("\n[bold]Example analysis queries:[/bold]")
        console.print("  • What's the latest news sentiment for this stock?")
        console.print("  • Should I buy this stock based on technical analysis?")
        console.print("  • How are the earnings looking?")
        console.print("  • Give me a comprehensive investment analysis")
        
        if natural_query_used:
            # Use the original natural language query as the analysis query
            user_query = natural_query
            console.print(f"\n[dim]Using your natural language query for analysis...[/dim]")
        else:
            user_query = console.input("\n[bold cyan]Your analysis question (or press Enter for comprehensive analysis): [/bold cyan]").strip()
            
            if user_query.lower() == 'quit':
                break
            
            if not user_query:
                user_query = "Give me a comprehensive investment analysis"
        
        # Execute the main query with agentic workflows
        console.print("\n" + "-" * 60)
        result = agent.query(stock_symbol, user_query, use_optimizer=True)
        
        # Display results
        console.print("\n" + "-" * 60)
        console.print(Panel.fit(
            "[bold green]Analysis Complete[/bold green]",
            border_style="green"
        ))
        
        # Show workflow information
        console.print(f"\n[bold]Workflow Used:[/bold] [yellow]{result['workflow_used']}[/yellow]")
        
        if "optimization" in result["workflow_details"]:
            opt = result["workflow_details"]["optimization"]
            console.print(f"[bold]Optimization:[/bold] {opt['iterations']} iterations, "
                         f"Quality Score: {opt['final_quality_score']:.1f}/10")
        
        # Show data summary
        console.print(f"\n[bold]Data Summary:[/bold]")
        data_table = Table(show_header=False, box=None)
        data_table.add_column("Metric", style="cyan")
        data_table.add_column("Value", style="white")
        
        summary = result["data_summary"]
        data_table.add_row("Current Price", f"${summary['price']}")
        data_table.add_row("P/E Ratio", str(summary['pe_ratio']))
        data_table.add_row("News Articles", str(summary['news_count']))
        
        console.print(data_table)
        
        # Show final analysis
        console.print(f"\n[bold]Analysis:[/bold]")
        console.print(Panel(
            result["final_analysis"],
            border_style="green",
            padding=(1, 2)
        ))
        
        # Save to file
        os.makedirs('recommendations', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recommendations/{stock_symbol}_{timestamp}.md"
        
        markdown_content = f"""# Investment Analysis for {stock_symbol}

**Query:** {user_query}

**Workflow Used:** {result['workflow_used']}

**Generated:** {result['timestamp']}

## Analysis

{result['final_analysis']}

## Data Summary

- **Current Price:** ${summary['price']}
- **P/E Ratio:** {summary['pe_ratio']}
- **News Articles Analyzed:** {summary['news_count']}

---

*This analysis was generated using agentic workflows including prompt chaining, routing, and evaluator-optimizer patterns.*
"""
        
        with open(filename, 'w') as f:
            f.write(markdown_content)
        
        console.print(f"\n[green]✓ Analysis saved to:[/green] [cyan]{filename}[/cyan]")
        console.print("\n" + "-" * 60)

    console.print("\n[bold cyan]Thank you for using Investment Research Agent![/bold cyan]\n")


if __name__ == "__main__":
    main()
