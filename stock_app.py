import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from ta import momentum, trend, volatility, volume
from transformers import pipeline
import torch

def get_basic_info(ticker):
    """Fetch basic stock information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    basic_info = {
        "Name": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": info.get("marketCap"),
        "Previous Close": info.get("previousClose"),
        "Open": info.get("open"),
    }
    return basic_info

def calculate_average_volume(ticker, days_list):
    """Calculate average volume traded over specified periods."""
    stock = yf.Ticker(ticker)
    today = datetime.today()
    avg_volumes = {}

    for days in days_list:
        start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
        data = stock.history(start=start_date)
        avg_volumes[f"{days} days"] = data["Volume"].mean()

    return avg_volumes

def get_company_overview(ticker):
    """Fetch company overview and recent performance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Get 52-week performance
    hist = stock.history(period="1y")
    year_high = hist['High'].max()
    year_low = hist['Low'].min()
    current_price = hist['Close'][-1]
    
    overview = {
        "Business Summary": info.get("longBusinessSummary"),
        "52 Week High": year_high,
        "52 Week Low": year_low,
        "Distance from High": f"{((year_high - current_price) / year_high * 100):.2f}%",
        "Distance from Low": f"{((current_price - year_low) / year_low * 100):.2f}%",
        "Website": info.get("website"),
        "Full Time Employees": info.get("fullTimeEmployees"),
    }
    return overview

def get_pe_comparison(ticker):
    """Get PE ratio comparison with sector."""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    pe_data = {
        "Company P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Sector P/E": info.get("trailingPE", 0) * 1.1,  # Approximation as YF doesn't provide sector PE
        "Industry P/E": info.get("trailingPE", 0) * 1.05,  # Approximation
    }
    return pe_data

def get_recent_news(ticker):
    """Fetch recent news about the company."""
    stock = yf.Ticker(ticker)
    news = stock.news[:3]  # Get last 3 news items
    
    formatted_news = []
    for item in news:
        formatted_news.append({
            "Title": item.get("title"),
            "Publisher": item.get("publisher"),
            "Date": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime('%Y-%m-%d'),
            "Link": item.get("link")
        })
    
    return formatted_news

def compare_performance(stock_ticker, index_ticker, periods):
    """Compare stock performance with a market index over specified periods."""
    results = []
    today = datetime.today()

    for days in periods:
        start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
        
        stock_data = yf.Ticker(stock_ticker).history(start=start_date)["Close"]
        index_data = yf.Ticker(index_ticker).history(start=start_date)["Close"]

        stock_return = (stock_data.iloc[-1] - stock_data.iloc[0]) / stock_data.iloc[0] * 100
        index_return = (index_data.iloc[-1] - index_data.iloc[0]) / index_data.iloc[0] * 100

        results.append({
            "Period (days)": days,
            "Stock Return (%)": stock_return,
            "Index Return (%)": index_return
        })

    return pd.DataFrame(results)

def get_fundamental_analysis(ticker):
    """Fetch detailed fundamental analysis data for a stock."""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    fundamental_data = {
        "Valuation Metrics": {
            "P/E Ratio": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "P/B Ratio": info.get("priceToBook"),
            "PEG Ratio": info.get("pegRatio"),
            "Enterprise Value/EBITDA": info.get("enterpriseToEbitda")
        },
        "Profitability Metrics": {
            "Gross Margin": info.get("grossMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Net Profit Margin": info.get("profitMargins"),
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets")
        },
        "Growth Metrics": {
            "Revenue Growth": info.get("revenueGrowth"),
            "Earnings Growth": info.get("earningsGrowth"),
            "Free Cash Flow Growth": info.get("freeCashflow"),
        },
        "Financial Health": {
            "Current Ratio": info.get("currentRatio"),
            "Debt to Equity": info.get("debtToEquity"),
            "Quick Ratio": info.get("quickRatio"),
            "Interest Coverage": info.get("interestCoverage")
        }
    }
    return fundamental_data

def get_technical_analysis(ticker, period="6mo"):
    """Calculate technical indicators for the stock."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = momentum.rsi(df['Close'], window=14)
    df['MACD'] = trend.macd_diff(df['Close'])
    df['BB_upper'] = volatility.bollinger_hband(df['Close'])
    df['BB_lower'] = volatility.bollinger_lband(df['Close'])
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    technical_signals = {
        "Price Trends": {
            "Above 20-day SMA": df['Close'].iloc[-1] > df['SMA_20'].iloc[-1],
            "Above 50-day SMA": df['Close'].iloc[-1] > df['SMA_50'].iloc[-1],
            "20-day SMA Trend": "Bullish" if df['SMA_20'].iloc[-1] > df['SMA_20'].iloc[-2] else "Bearish"
        },
        "Momentum": {
            "RSI": df['RSI'].iloc[-1],
            "RSI Signal": "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral",
            "MACD Signal": "Bullish" if df['MACD'].iloc[-1] > 0 else "Bearish"
        },
        "Volatility": {
            "Above Upper BB": df['Close'].iloc[-1] > df['BB_upper'].iloc[-1],
            "Below Lower BB": df['Close'].iloc[-1] < df['BB_lower'].iloc[-1],
            "BB Position": "Outside Upper" if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1] else "Outside Lower" if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1] else "Inside Bands"
        }
    }
    
    return df, technical_signals


def generate_analysis_text(technical_data, fundamental_data):
    """Generate text for LLM analysis based on technical and fundamental indicators."""
    analysis_text = f"""
    Based on the technical analysis:
    - RSI is {technical_data['Momentum']['RSI']:.2f} ({technical_data['Momentum']['RSI Signal']})
    - MACD signal is {technical_data['Momentum']['MACD Signal']}
    - Price trend: {technical_data['Price Trends']['20-day SMA Trend']}
    - Stock is trading {'above' if technical_data['Price Trends']['Above 20-day SMA'] else 'below'} 20-day SMA
    - Stock is trading {'above' if technical_data['Price Trends']['Above 50-day SMA'] else 'below'} 50-day SMA
    
    Key fundamental metrics:
    - P/E Ratio: {fundamental_data['Valuation Metrics']['P/E Ratio']}
    - PEG Ratio: {fundamental_data['Valuation Metrics']['PEG Ratio']}
    - Debt to Equity: {fundamental_data['Financial Health']['Debt to Equity']}
    - Revenue Growth: {fundamental_data['Growth Metrics']['Revenue Growth']}
    
    Based on these indicators, provide an investment recommendation including:
    1. Whether to invest or not
    2. If yes, suggest a long-term investment strategy
    3. Key risks to monitor
    4. Suggested position sizing and time horizon
    """
    return analysis_text

from huggingface_hub import InferenceClient
def get_response(prompt):
    
    client = InferenceClient(api_key="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")
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct", 
        messages=messages, 
        max_tokens=2500
    )

    return completion.choices[0].message.content

def get_ai_recommendation(technical_signals, fundamental_data):
    """Get AI-powered investment recommendation."""
    try:
        analysis_text = generate_analysis_text(technical_signals, fundamental_data)
        response = get_response(analysis_text)
        return response
    except Exception as e:
        return f"Error generating AI recommendation: {str(e)}\n\nPlease try again later or consider the technical and fundamental analysis separately."

# Streamlit App
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Fundamental Analysis", "Technical Analysis", "AI Recommendation", "Strategy"])

# Sidebar for user input
st.sidebar.header("Input Parameters")
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
index_ticker = st.sidebar.text_input("Enter Index Ticker (e.g., ^NSEI for Nifty 50):", "^NSEI")
days_list = [3, 5, 10, 30]
comparison_periods = [3, 5, 10, 30, 60]

if st.sidebar.button("Analyze"):
    # Get all the required data upfront
    basic_info = get_basic_info(stock_ticker)
    overview = get_company_overview(stock_ticker)
    fundamental_data = get_fundamental_analysis(stock_ticker)
    df, technical_signals = get_technical_analysis(stock_ticker)

    # Tab 1: Fundamental Analysis
    with tab1:
        st.header("Company Overview  - " + stock_ticker)
        st.table(pd.DataFrame([basic_info]))
        st.write("### Business Summary")
        st.write(overview["Business Summary"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("52 Week Performance")
            performance_data = {
                "52 Week High": overview["52 Week High"],
                "52 Week Low": overview["52 Week Low"],
                "Distance from High": overview["Distance from High"],
                "Distance from Low": overview["Distance from Low"]
            }
            st.table(pd.DataFrame([performance_data]))
            
            st.subheader("Average Volume Traded")
            avg_volumes = calculate_average_volume(stock_ticker, days_list)
            avg_volume_df = pd.DataFrame(list(avg_volumes.items()), columns=["Period", "Average Volume"])
            st.table(avg_volume_df)
            
        with col2:
            st.subheader("P/E Comparison")
            pe_data = get_pe_comparison(stock_ticker)
            st.table(pd.DataFrame([pe_data]))
            
            st.subheader("Company Information")
            company_data = {
                "Website": overview["Website"],
                "Employees": overview["Full Time Employees"]
            }
            st.table(pd.DataFrame([company_data]))
        
        # st.subheader("Latest News")
        # news = get_recent_news(stock_ticker)
        # for item in news:
        #     st.write(f"**{item['Title']}**")
        #     st.write(f"Publisher: {item['Publisher']} | Date: {item['Date']}")
        #     st.write(f"[Read More]({item['Link']})")
        #     st.write("---")
        
        st.header("Detailed Fundamental Analysis")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Valuation Metrics")
            st.table(pd.DataFrame([fundamental_data["Valuation Metrics"]]))
            
            st.subheader("Profitability Metrics")
            st.table(pd.DataFrame([fundamental_data["Profitability Metrics"]]))
            
        with col4:
            st.subheader("Growth Metrics")
            st.table(pd.DataFrame([fundamental_data["Growth Metrics"]]))
            
            st.subheader("Financial Health")
            st.table(pd.DataFrame([fundamental_data["Financial Health"]]))

    # Tab 2: Technical Analysis
    with tab2:
        st.header("Technical Analysis")
        
        st.subheader("Price and Moving Averages")
        chart_data = pd.DataFrame({
            'Close': df['Close'],
            '20-day SMA': df['SMA_20'],
            '50-day SMA': df['SMA_50']
        })
        st.line_chart(chart_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Trends")
            st.table(pd.DataFrame([technical_signals["Price Trends"]]))
            
            st.subheader("Momentum Indicators")
            st.table(pd.DataFrame([technical_signals["Momentum"]]))
            
        with col2:
            st.subheader("RSI")
            st.line_chart(df['RSI'])
            
            st.subheader("MACD")
            st.line_chart(df['MACD'])
            
        st.subheader("Bollinger Bands")
        bb_data = pd.DataFrame({
            'Price': df['Close'],
            'Upper Band': df['BB_upper'],
            'Lower Band': df['BB_lower']
        })
        st.line_chart(bb_data)

    # Tab 3: AI Recommendation
    with tab3:
        st.header("AI-Powered Investment Analysis")
        
        with st.spinner("Generating AI recommendation..."):
            recommendation = get_ai_recommendation(technical_signals, fundamental_data)
        
        st.subheader("Investment Recommendation")
        st.write(recommendation)
        
        st.subheader("Analysis Confidence Metrics")
        
        # Calculate confidence scores based on technical indicators
        technical_confidence = 0
        if 40 <= technical_signals['Momentum']['RSI'] <= 60:
            technical_confidence += 0.3
        if technical_signals['Price Trends']['Above 20-day SMA']:
            technical_confidence += 0.2
        if technical_signals['Price Trends']['Above 50-day SMA']:
            technical_confidence += 0.2
        if technical_signals['Momentum']['MACD Signal'] == "Bullish":
            technical_confidence += 0.3
            
        # Calculate fundamental confidence
        fundamental_confidence = 0
        if fundamental_data['Valuation Metrics']['P/E Ratio'] and fundamental_data['Valuation Metrics']['P/E Ratio'] > 0:
            fundamental_confidence += 0.25
        if fundamental_data['Growth Metrics']['Revenue Growth'] and fundamental_data['Growth Metrics']['Revenue Growth'] > 0:
            fundamental_confidence += 0.25
        if fundamental_data['Financial Health']['Debt to Equity'] and fundamental_data['Financial Health']['Debt to Equity'] < 2:
            fundamental_confidence += 0.25
        if fundamental_data['Profitability Metrics']['Net Profit Margin'] and fundamental_data['Profitability Metrics']['Net Profit Margin'] > 0:
            fundamental_confidence += 0.25
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Technical Analysis Confidence", f"{technical_confidence:.0%}")
        with col2:
            st.metric("Fundamental Analysis Confidence", f"{fundamental_confidence:.0%}")
            
        st.warning("""
        **Disclaimer**: This AI-powered analysis is for informational purposes only and should not be 
        considered as financial advice. Always conduct your own research and consult with a qualified 
        financial advisor before making investment decisions.
        """)

    # Tab 4: Strategy
    with tab4:
        st.header("Investment Strategy Guidelines")
        
        # Calculate overall market conditions
        market_condition = "Neutral"
        if technical_signals['Momentum']['RSI'] > 70:
            market_condition = "Overbought"
        elif technical_signals['Momentum']['RSI'] < 30:
            market_condition = "Oversold"
            
        risk_level = "Moderate"
        if fundamental_data['Financial Health']['Debt to Equity'] and fundamental_data['Financial Health']['Debt to Equity'] > 2:
            risk_level = "High"
        elif fundamental_data['Financial Health']['Debt to Equity'] and fundamental_data['Financial Health']['Debt to Equity'] < 1:
            risk_level = "Low"
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Conditions")
            st.write(f"Current Market Condition: **{market_condition}**")
            st.write(f"Risk Level: **{risk_level}**")
            
            st.subheader("Position Sizing Recommendation")
            if risk_level == "High":
                st.write("Consider smaller position size (1-2% of portfolio)")
            elif risk_level == "Low":
                st.write("Standard position size (3-5% of portfolio)")
            else:
                st.write("Moderate position size (2-3% of portfolio)")
                
        with col2:
            st.subheader("Investment Horizon")
            st.write("Recommended holding period based on analysis:")
            if market_condition == "Oversold" and risk_level == "Low":
                st.write("- Long-term investment (3+ years)")
            elif market_condition == "Overbought":
                st.write("- Consider waiting for better entry point")
            else:
                st.write("- Medium-term investment (1-3 years)")
                
        st.subheader("Risk Management Guidelines")
        st.write("""
        1. Set stop-loss orders 10-15% below purchase price
        2. Consider dollar-cost averaging for entering positions
        3. Review position quarterly or when technical signals change significantly
        4. Monitor key technical levels and fundamental metrics
        """)
        
        st.subheader("Exit Strategy")
        st.write("""
        Consider exiting or reducing position if:
        1. Technical indicators show significant deterioration
        2. Fundamental metrics decline for 2+ consecutive quarters
        3. Stop-loss levels are breached
        4. Investment thesis changes materially
        """)