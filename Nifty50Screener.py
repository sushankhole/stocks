from inspect import trace
from textwrap import fill
import pandas as pd
from pandas.core.indexes import period
from traitlets.traitlets import Unicode
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser



nft50 = pd.read_csv("NIFTY50.csv")
symbols = nft50['Symbol'].sort_values().tolist()        

ticker = st.sidebar.selectbox(
    'Choose a NIfty 50 Stock',
     symbols)

infoType = st.sidebar.radio(
        "Choose an info type",
        ('Fundamental', 'Technical')
    )
img = st.image('Nifty_50_Logo.svg')

stock = yf.Ticker(ticker)

if(infoType == 'Fundamental'):
    stock = yf.Ticker(ticker)
    info = stock.info 
    st.title('Company Profile')
    st.subheader(info['longName']) 
    st.markdown('** Sector **: ' + info['sector'])
    st.markdown('** Industry **: ' + info['industry'])
    st.markdown('** Phone **: ' + info['phone'])
    st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
    st.markdown('** Website **: ' + info['website'])
    st.markdown('** Business Summary **')
    st.info(info['longBusinessSummary'])

    fundInfo = {
        'Enterprise Value (INR) ' : info['enterpriseValue'],
        'Enterprise To Reevenue Ratio' : info['enterpriseToRevenue'],
        'Enterprise To Ebitda Ratio' : info['enterpriseToEbitda'],
        'Net Income (INR)' : info['netIncomeToCommon'],
        'Profit Margin Ratio' : info['profitMargins'],
        'Forward PE Ratio' : info['forwardPE'],
        'PEG Ratio' : info['pegRatio'],
        'Price to Book Ratio' : info['priceToBook'],
        'Price to Book Ratio' : info['priceToBook'],
        'Forward EPS (INR)' : info['forwardEps'],
        'Beta ' : info['beta'],
        'Book Value (INR)' : info['bookValue'],
        'Dividend Rate (%)' : info['dividendRate'], 
        'Dividend Yield (%)' : info['dividendYield'],
        'Five year Avg Dividend Yield (%)' :  info['fiveYearAvgDividendYield'],
        'Payout Ratio' : info['payoutRatio']
    }

    fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
    fundDF = fundDF.rename(columns={0: 'Value'})
    st.subheader('Fundamental Info')
    st.table(fundDF)

    st.subheader('General Stock Info')
    st.markdown('** Market **: ' +  info['market'])
    st.markdown('** Exchange **: ' +  info['exchange'])
    st.markdown('** Quote Day **: ' +  info['quoteType'])

    num_year = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2)

    start = dt.datetime.today()-dt.timedelta(num_year * 365)
    end = dt.datetime.today()
    df = yf.download(ticker,start,end)
    df = df.reset_index()
    fig = go.Figure(
                data=go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))

    fig.update_layout(
                title={
                    'text': 'Stock Prices of '+ ticker + ' Over Last ' + str(num_year) + ' Year(s)',
                    'y': 0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    fig.update_yaxes(tickprefix='₹')


    marketInfo = {
            "Volume": info['volume'],
            "Average Volume": info['averageVolume'],
            "Market Cap": info["marketCap"],
            "Float Shares": info['floatShares'],
            "Regular Market Price (INR)": info['regularMarketPrice'],
            'Share Outstanding': info['sharesOutstanding']
    
        }
    
    marketDF = pd.DataFrame(data=marketInfo,index=[1])
    st.table(marketDF)
else:
    def calc_moving_average(data,size):
        df = data.copy()
        df['sma'] = df['Close'].rolling(int(size)).mean()
        df['ema'] = df['Close'].ewm(span=size,min_periods=size).mean()
        df.dropna(inplace=True)
        return df

    def calc_macd(data):
        df = data.copy()
        df['ema12'] = df['Close'].ewm(span=12,min_periods=12).mean()
        df['ema26'] = df['Close'].ewm(span=26,min_periods=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
        df.dropna(inplace=True)
        return df

    def calc_bollinger(data,size):
        df = data.copy()
        df['sma'] = df['Close'].rolling(int(size)).mean()
        df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(int(size)).std(ddof=0) 
        df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(int(size)).std(ddof=0) 
        df["width"] = df["bolu"] - df["bold"]
        df.dropna(inplace=True)
        return df

    st.title('Technical Indicators')
    st.subheader('Moving Average')
    coMA1, coMA2 = st.beta_columns(2)

    with coMA1:
        num_year_MA = st.number_input('Insert period(Year):', min_value=1,max_value=10,value=2,key='1')

    with coMA2:
        window_size_MA = st.number_input('Window Size(Day):', min_value=5,max_value=500,value=20,key='2')

    start = dt.datetime.today()-dt.timedelta(num_year_MA*365)
    end = dt.datetime.today()
    dataMA = yf.download(ticker,start,end)
    df_ma = calc_moving_average(dataMA,window_size_MA)
    df_ma = df_ma.reset_index()

    figMA = go.Figure()
    figMA.add_trace(
        go.Scatter(
            x = df_ma['Date'],
            y = df_ma['Close'],
            name = "Prices over last " + str(num_year_MA) + 'Year(s)' 
        )
    )

    figMA.add_trace(
        go.Scatter(
            x = df_ma['Date'],
            y = df_ma['sma'],
            name = 'SMA ' + str(window_size_MA) + ' Over Last' + str(num_year_MA) + 'Year(s)'   
        )
    )

    figMA.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma['ema'],
                name = 'EMA ' + str(window_size_MA) + ' Over Last' + str(num_year_MA) + 'Year(s)'

            )
    )

    figMA.update_layout(legend=dict(
        yanchor='top',
        y=0.99,
        xanchor='left',
        x=0.01
    ))

    figMA.update_layout(legend_title_text='Trend')
    figMA.update_yaxes(tickprefix='₹ ')

    st.plotly_chart(figMA, use_container_width=True)

    st.subheader("Moving Average Convergance Divergence (MACD) ")
    num_year_MACD = st.number_input('Insert period(Year):', min_value=1,max_value=10,value=2,key='3')

    start_MACD = dt.datetime.today()-dt.timedelta(num_year_MACD*365)
    end_MACD = dt.datetime.today()
    dataMACD = yf.download(ticker,start_MACD,end_MACD)
    df_macd = calc_macd(dataMACD)
    df_macd = df_macd.reset_index()

    figMACD = make_subplots(rows=2,cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    figMACD.add_trace(
        go.Scatter(
            x = df_macd['Date'],
            y = df_macd['Close'],
            name = "Prices Over Last " + str(num_year_MACD) + 'Year(s)'
        )
    )

    figMACD.add_trace(
        go.Scatter(
            x = df_macd['Date'],
            y = df_macd['ema12'],
            name = 'EMA 12 Over Last ' + str(num_year_MACD) + 'Year(s)'
        ),
        row=1,col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x = df_macd['Date'],
            y = df_macd['ema26'],
            name = 'EMA 12 Over Last ' + str(num_year_MACD) + ' Years(s)'

        ),
        row=1,col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x = df_macd['Date'],
            y = df_macd['macd'],
            name='MACD Line'
        ),
        row=2,col=1
    )

    figMACD.add_trace(
        go.Scatter(
            x = df_macd['Date'],
            y = df_macd['signal'],
            name  = 'Signal Line'
        ),
        row=2,col=1
    )

    figMACD.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0
    )

    )

    figMACD.update_yaxes(tickprefix='₹')
    st.plotly_chart(figMACD, use_container_width=True)

    st.subheader('Bollinger Band')
    coBoll1, coBoll2 = st.beta_columns(2)

    with coBoll1:
        num_year_boll = st.number_input('Insert Period (Year): ', min_value=1, max_value=10,value=2,key='6')

    with coBoll2:
        window_size_boll = st.number_input('Window Size (Day): ',min_value=5, max_value=500, value=20, key='7')

    start_boll = dt.datetime.today()-dt.timedelta(num_year_boll*365)
    end_boll = dt.datetime.today() 
    dataBoll = yf.download(ticker,start_boll,end_boll)
    df_boll = calc_bollinger(dataBoll,window_size_boll)
    df_boll = df_boll.reset_index()
    figBoll = go.Figure()
    figBoll.add_trace(
        go.Scatter(
            x = df_boll['Date'],
            y = df_boll['bolu'],
            name = 'Upper Band'
        )
    )

    figBoll.add_trace(
        go.Scatter(
            x = df_boll['Date'],
            y = df_boll['sma'],
            name = 'SMA ' + str(window_size_boll) + ' Over Last ' + str(num_year_boll) + ' Year(s)'

        )
    )

    figBoll.add_trace(
        go.Scatter(
            x = df_boll['Date'],
            y = df_boll['bold'],
            name = "Lower Band"
        )
    )

    figBoll.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1,
        xanchor='left',
        x=0
    )
    )

    figBoll.update_yaxes(tickprefix='₹')
    st.plotly_chart(figBoll, use_container_width=True)

    st.header("FAQ:")
    st.subheader('1. What is Moving Average?')
    st.write('In finance, a moving average (MA) is a stock indicator that is commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price.')
    st.subheader('2. What is Simple Moving Average(SMA)?')
    st.write('The simplest form of a moving average, known as a simple moving average (SMA), is calculated by taking the arithmetic mean of a given set of values. In other words, a set of numbers–or prices in the case of financial instruments–are added together and then divided by the number of prices in the set.')
    st.subheader('3. What is Exponential Moving Average (EMA)?')
    st.write('The exponential moving average is a type of moving average that gives more weight to recent prices in an attempt to make it more responsive to new information.')
    st.subheader('4. What is Moving Average Convergance Divergence (MACD)?')
    st.write('Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals. Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line. Moving average convergence divergence (MACD) indicators can be interpreted in several ways, but the more common methods are crossovers, divergences, and rapid rises/falls.')  
    st.subheader('5. What is Bollinger Band? ')
    st.write("A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences.")
    st.subheader('6. Who are you? Why did you develop this project?')
    st.write('Just a final-year IT student who loves machine learning and I did this project as part of my study. Thank you!! ')
    st.write('My Accounts:')
    url = 'https://www.linkedin.com/in/sumit-redekar-101817205/'
    if st.button("LinkedIn"):
        webbrowser.open_new_tab(url)
    url1 = 'https://github.com/sumittttttt'
    if st.button("Github"):
        webbrowser.open_new_tab(url1)
