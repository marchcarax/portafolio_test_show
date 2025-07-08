import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.mc_completed = False

# Page config
st.set_page_config(page_title="Neoangelo Portfolio Analytics", layout="wide")

# Title
st.title("📊 Advanced Neoangelo Portfolio Analytics")
st.markdown("---")

# Vantage portfolio configuration
VANTAGE_PORTFOLIO = {
    'USDJPY':     {'lots': -0.05, 'notional': 4264,  'weight': 0.3011, 'yf_ticker': 'USDJPY=X'},
    'XAUUSD':     {'lots': 0.01,  'notional': 3304,  'weight': 0.2334, 'yf_ticker': 'GC=F'},
    'SOYBEANS':   {'lots': 0.1,   'notional': 432,   'weight': 0.0305, 'yf_ticker': 'ZS=F'},
    'NVDA':       {'lots': -3,    'notional': 408,   'weight': 0.0288, 'yf_ticker': 'NVDA'},
    'ABNB':       {'lots': -5,    'notional': 584,   'weight': 0.0412, 'yf_ticker': 'ABNB'},
    'GOOGL':      {'lots': 5,     'notional': 744,   'weight': 0.0525, 'yf_ticker': 'GOOGL'},
    'GS':         {'lots': -1,    'notional': 595,   'weight': 0.0420, 'yf_ticker': 'GS'},
    'HOOD':       {'lots': 6,     'notional': 471,   'weight': 0.0333, 'yf_ticker': 'HOOD'},
    'JPM':        {'lots': 6,     'notional': 1446,  'weight': 0.1021, 'yf_ticker': 'JPM'},
    'META':       {'lots': 1,     'notional': 612,   'weight': 0.0432, 'yf_ticker': 'META'},
    'MS':         {'lots': 5,     'notional': 603,   'weight': 0.0426, 'yf_ticker': 'MS'},
    'RBLX':       {'lots': 5,     'notional': 446,   'weight': 0.0315, 'yf_ticker': 'RBLX'},
    'UBER':       {'lots': 3,     'notional': 247,   'weight': 0.0174, 'yf_ticker': 'UBER'}
}

# Default pairs trading
DEFAULT_PAIRS = [
    {'long': 'XAUUSD', 'short': 'XAGUSD'},
    {'long': 'QQQ', 'short': 'IWM'},
]

# Helper functions
@st.cache_data
def fetch_data(portfolio_config, start_date, end_date):
    """Fetch data from yfinance using portfolio configuration"""
    prices = pd.DataFrame()
    ticker_map = {}
    
    for asset, config in portfolio_config.items():
        try:
            yf_ticker = config['yf_ticker']
            ticker_map[asset] = yf_ticker
            
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                prices[asset] = df['Close']
        except Exception as e:
            st.warning(f"Could not fetch data for {asset} ({yf_ticker}): {str(e)}")
    
    return prices, ticker_map

def calculate_position_values(portfolio_config, prices):
    """Calculate position values based on lots and current prices"""
    positions = pd.DataFrame(index=prices.index)
    lot_sizes = {}
    
    for asset in prices.columns:  # Only process assets that exist in price data
        if asset in portfolio_config:
            config = portfolio_config[asset]
            # Calculate position value: lots * notional_per_lot
            # notional_per_lot = notional_value / abs(lots)
            notional_per_lot = config['notional'] / abs(config['lots']) if config['lots'] != 0 else 0
            positions[asset] = config['lots'] * notional_per_lot
            lot_sizes[asset] = notional_per_lot
    
    return positions, lot_sizes

def calculate_portfolio_metrics(returns, positions, total_capital):
    """Calculate comprehensive portfolio metrics"""
    # Ensure positions and returns have the same index
    common_index = returns.index.intersection(positions.index)
    returns_aligned = returns.loc[common_index]
    positions_aligned = positions.loc[common_index]
    
    # Portfolio returns
    weighted_returns = (returns_aligned * positions_aligned).div(total_capital)
    portfolio_returns = weighted_returns.sum(axis=1)
    
    # Net exposure - get the last row values
    net_exposure = positions.iloc[-1].sum() / total_capital
    gross_exposure = positions.iloc[-1].abs().sum() / total_capital
    
    # Performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # VaR (95% and 99%)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    # CVaR
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    # Profit factor
    gains = portfolio_returns[portfolio_returns > 0].sum()
    losses = abs(portfolio_returns[portfolio_returns < 0].sum())
    profit_factor = gains / losses if losses != 0 else np.inf
    
    # Gain to pain ratio
    gain_to_pain = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    # Sortino ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annual_return / downside_std if downside_std != 0 else np.inf
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'profit_factor': profit_factor,
        'gain_to_pain': gain_to_pain,
        'calmar_ratio': calmar_ratio,
        'net_exposure': net_exposure,
        'gross_exposure': gross_exposure
    }

def calculate_risk_parity_weights(returns, current_weights):
    """Calculate risk parity weights and compare with current allocation"""
    # Covariance matrix
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Current risk contribution
    portfolio_vol = np.sqrt(current_weights @ cov_matrix @ current_weights)
    marginal_contrib = cov_matrix @ current_weights
    contrib = current_weights * marginal_contrib
    risk_contrib = contrib / portfolio_vol
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100
    
    # Risk parity optimization (equal risk contribution)
    n_assets = len(returns.columns)
    target_risk_contrib = 1 / n_assets
    
    # Simple risk parity approximation
    asset_vols = np.sqrt(np.diag(cov_matrix))
    rp_weights = (1 / asset_vols) / (1 / asset_vols).sum()
    
    return {
        'current_weights': current_weights,
        'current_risk_contrib': risk_contrib_pct,
        'rp_weights': rp_weights * 100,
        'asset_vols': asset_vols * 100
    }

# Sidebar inputs
st.sidebar.header("Analysis Configuration")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Portfolio setup - Fixed to Vantage Portfolio
portfolio_config = VANTAGE_PORTFOLIO.copy()
total_capital = sum(config['notional'] for config in portfolio_config.values())

st.sidebar.success("Using Vantage Portfolio")
st.sidebar.metric("Total Portfolio Value", f"${total_capital:,.0f}")

# Pairs trading configuration
st.sidebar.subheader("Pairs Trading Configuration")
use_default_pairs = st.sidebar.checkbox("Use Default Pairs", value=True)

if use_default_pairs:
    pairs_config = DEFAULT_PAIRS
    st.sidebar.success("Using default pairs configuration")
    with st.sidebar.expander("View Default Pairs"):
        for i, pair in enumerate(DEFAULT_PAIRS):
            st.write(f"Pair {i+1}: {pair['long']} / {pair['short']}")
else:
    st.sidebar.info("Custom pairs configuration disabled for this version")
    pairs_config = DEFAULT_PAIRS

# Analyze button
if st.sidebar.button("Analyze Portfolio", type="primary"):
    with st.spinner("Fetching data..."):
        prices, ticker_map = fetch_data(portfolio_config, start_date, end_date)
        
        if prices.empty:
            st.error("No data fetched. Please check your configuration.")
        else:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate positions based on lots and notional values
            positions, lot_sizes = calculate_position_values(portfolio_config, prices)
            
            # Store in session state
            st.session_state['prices'] = prices
            st.session_state['returns'] = returns
            st.session_state['positions'] = positions
            st.session_state['portfolio_config'] = portfolio_config
            st.session_state['total_capital'] = total_capital
            st.session_state['pairs_config'] = pairs_config
            st.session_state['lot_sizes'] = lot_sizes

# Main content
if 'prices' in st.session_state:
    prices = st.session_state['prices']
    returns = st.session_state['returns']
    positions = st.session_state['positions']
    portfolio_config = st.session_state.get('portfolio_config', VANTAGE_PORTFOLIO)
    total_capital = st.session_state.get('total_capital', sum(config['notional'] for config in VANTAGE_PORTFOLIO.values()))
    pairs_config = st.session_state.get('pairs_config', DEFAULT_PAIRS)
    lot_sizes = st.session_state.get('lot_sizes', {})
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(returns, positions, total_capital)
    
    # Create main tabs
    tab1, tab2 = st.tabs([
        "📊 Portfolio Analysis", 
        "⚖️ Risk & Optimization"
    ])
    
    with tab1:  # Portfolio Analysis Tab
        # Display portfolio composition
        st.header("💼 Portfolio Composition")
        
        comp_data = []
        for asset, config in portfolio_config.items():
            if asset in prices.columns:
                comp_data.append({
                    'Asset': asset,
                    'Lots': config['lots'],
                    'Notional Value': f"${config['notional']:,.0f}",
                    'Weight': f"{config['notional']/total_capital*100:.1f}%",
                    'Type': 'LONG' if config['lots'] > 0 else 'SHORT'
                })
        
        comp_df = pd.DataFrame(comp_data)
    
        # Style the dataframe
        def style_type(val):
            return 'color: green' if val == 'LONG' else 'color: red'
    
        styled_comp = comp_df.style.applymap(style_type, subset=['Type'])
        st.dataframe(styled_comp, use_container_width=True)
    
        # Display key metrics
        st.header("📈 Portfolio Performance Metrics")
    
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
            st.metric("Annual Return", f"{metrics['annual_return']*100:.2f}%")
        with col2:
            st.metric("Volatility", f"{metrics['volatility']*100:.2f}%")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
        with col4:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Gain/Pain Ratio", f"{metrics['gain_to_pain']:.2f}")
        with col5:
            st.metric("Net Exposure", f"{metrics['net_exposure']*100:.1f}%")
            st.metric("Gross Exposure", f"{metrics['gross_exposure']*100:.1f}%")
        
        st.markdown("---")
    
        # Risk metrics with explanations
        st.header("⚠️ Risk Metrics")
        
        # Add explanatory note
        with st.expander("📝 Risk Metrics Explanation"):
            st.markdown("""
            **Value at Risk (VaR)**: The maximum loss expected over a specific time period with a given confidence level.
            - VaR 95%: There's a 5% chance that daily losses will exceed this amount
            - VaR 99%: There's a 1% chance that daily losses will exceed this amount
            
            **Conditional Value at Risk (CVaR)**: The expected loss beyond the VaR threshold.
            - CVaR 95%: Average of all losses that exceed the 95% VaR
            - CVaR 99%: Average of all losses that exceed the 99% VaR
            
            CVaR provides insight into tail risk - how bad losses could be when they exceed the VaR threshold.
            """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
        with col2:
            st.metric("VaR (99%)", f"{metrics['var_99']*100:.2f}%")
        with col3:
            st.metric("CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
        with col4:
            st.metric("CVaR (99%)", f"{metrics['cvar_99']*100:.2f}%")
        
        # Portfolio returns chart
        st.header("📊 Portfolio Analysis")
    
        # Calculate portfolio value over time
        # Ensure alignment between returns and positions
        common_index = returns.index.intersection(positions.index)
        returns_aligned = returns.loc[common_index]
        positions_aligned = positions.loc[common_index]
        
        weighted_returns = (returns_aligned * positions_aligned).div(total_capital)
        portfolio_returns = weighted_returns.sum(axis=1)
        portfolio_value = total_capital * (1 + portfolio_returns).cumprod()
        
        # Create tabs for different charts
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["Portfolio Value", "Asset Performance", "Correlations", "Risk Analysis"])
    
        with chart_tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values,
                                    mode='lines', name='Portfolio Value',
                                    line=dict(color='blue', width=2)))
            fig.update_layout(title='Portfolio Value Over Time',
                            xaxis_title='Date', yaxis_title='Value ($)',
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown chart
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                    fill='tozeroy', name='Drawdown',
                                    line=dict(color='red', width=1)))
            fig_dd.update_layout(title='Portfolio Drawdown',
                                xaxis_title='Date', yaxis_title='Drawdown (%)',
                                height=300)
            st.plotly_chart(fig_dd, use_container_width=True)
    
        with chart_tab2:
            # Individual asset performance
            fig = go.Figure()
            for asset in prices.columns:
                normalized_price = prices[asset] / prices[asset].iloc[0] * 100
                fig.add_trace(go.Scatter(x=normalized_price.index, y=normalized_price.values,
                                        mode='lines', name=asset))
            fig.update_layout(title='Normalized Asset Performance (Base = 100)',
                            xaxis_title='Date', yaxis_title='Normalized Price',
                            height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_tab3:
            # Correlation matrix
            corr_matrix = returns.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r', 
                        labels=dict(color="Correlation"))
            fig.update_layout(title='Asset Correlation Matrix', height=600)
            st.plotly_chart(fig, use_container_width=True)
    
        with chart_tab4:
            # Risk contribution
            position_values = positions.iloc[-1]
            position_weights = position_values / total_capital
            
            # Calculate risk contribution (simplified)
            asset_vols = returns.std() * np.sqrt(252)
            risk_contribution = abs(position_weights) * asset_vols
            risk_contribution = risk_contribution / risk_contribution.sum() * 100
            
            fig = go.Figure(data=[
                go.Bar(x=risk_contribution.index, y=risk_contribution.values,
                    marker_color=['green' if portfolio_config.get(asset, {}).get('lots', 0) > 0 else 'red' 
                                for asset in risk_contribution.index])
            ])
            fig.update_layout(title='Risk Contribution by Asset (%)',
                            xaxis_title='Asset', yaxis_title='Risk Contribution (%)',
                            height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Risk Parity Analysis
        st.markdown("---")
        st.header("⚖️ Risk Parity & Portfolio Optimization")
        
        # Calculate current risk contributions
        current_weights = positions.iloc[-1] / total_capital
        risk_parity_analysis = calculate_risk_parity_weights(returns, current_weights)
        
        # Display risk contribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Risk Contribution")
            risk_contrib_df = pd.DataFrame({
                'Asset': returns.columns,
                'Weight (%)': current_weights.values * 100,
                'Risk Contrib (%)': risk_parity_analysis['current_risk_contrib'].values,
                'Volatility (%)': risk_parity_analysis['asset_vols']
            })
            
            # Highlight over/under allocated
            def highlight_risk(row):
                target = 100 / len(returns.columns)
                if row['Risk Contrib (%)'] > target * 1.5:
                    return ['background-color: #FFB6C1'] * len(row)
                elif row['Risk Contrib (%)'] < target * 0.5:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            styled_risk_df = risk_contrib_df.style.apply(highlight_risk, axis=1).format({
                'Weight (%)': '{:.1f}',
                'Risk Contrib (%)': '{:.1f}',
                'Volatility (%)': '{:.1f}'
            })
            st.dataframe(styled_risk_df, use_container_width=True)
        
        with col2:
            # Risk contribution pie chart
            fig = px.pie(risk_contrib_df, values='Risk Contrib (%)', names='Asset',
                        title='Risk Contribution by Asset')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please click 'Analyze Portfolio' in the sidebar to begin analysis.")
    
    st.markdown("""
    ### 📋 Instructions:
    
    1. **Configure Date Range**: Select the analysis period in the sidebar
    2. **Pairs Trading**: Toggle to use default pairs or configure custom pairs
    3. **Click 'Analyze Portfolio'** to generate comprehensive analysis
    
    ### 🎯 Vantage Portfolio Includes:
    - **Forex**: AUDJPY (Long), EURCHF (Short), USDJPY (Short), EURJPY (Short), GBPJPY (Long)
    - **Indices**: GER40/DAX (Long), SP500 (Long)
    - **Bonds**: USNOTE10YR (Long)
    - **Commodities**: XAUUSD/Gold (Long), XAGUSD/Silver (Short), UKOIL (Long)
    
    ### 🔄 Default Pairs:
    - **XAUUSD/XAGUSD**: Gold vs Silver spread
    - **QQQ/IWM**: Tech vs Small Cap spread
    
    ### 📈 Analysis Features:
    - **Performance Metrics**: Sharpe, Sortino, Profit Factor, Calmar Ratio
    - **Risk Analysis**: VaR, CVaR, Maximum Drawdown with explanations
    - **Position Analysis**: Long/Short breakdown with weights
    - **Visualization**: Portfolio value, drawdowns, correlations, risk contribution
    - **Risk Parity**: Current vs optimal risk allocation
    """)

# Footer
st.markdown("---")
st.markdown("*Note: This is for educational purposes. Always do your own research before making investment decisions.*")