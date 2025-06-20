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

# Darwin portfolio configuration (formerly DEFAULT_PORTFOLIO)
# Portfolio configurations

VANTAGE_PORTFOLIO = {
    'AUDJPY': {'lots': 0.02, 'notional': 1126, 'weight': 0.04, 'yf_ticker': 'AUDJPY=X'},
    'EURCHF': {'lots': -0.02, 'notional': 2000, 'weight': 0.06, 'yf_ticker': 'EURCHF=X'},
    'GER40': {'lots': 0.2, 'notional': 4723, 'weight': 0.15, 'yf_ticker': '^GDAXI'},
    'SP500': {'lots': 1, 'notional': 5188, 'weight': 0.16, 'yf_ticker': 'SPY'},
    'UKOIL': {'lots': 0.01, 'notional': 640, 'weight': 0.02, 'yf_ticker': 'BZ=F'},
    'USDJPY': {'lots': -0.03, 'notional': 2560, 'weight': 0.08, 'yf_ticker': 'USDJPY=X'},
    'USNOTE10YR': {'lots': 30, 'notional': 2863, 'weight': 0.09, 'yf_ticker': '^TNX'},
    'XAGUSD': {'lots': -0.01, 'notional': 1807, 'weight': 0.06, 'yf_ticker': 'SI=F'},
    'XAUUSD': {'lots': 0.02, 'notional': 6865, 'weight': 0.21, 'yf_ticker': 'GC=F'},
    'EURJPY': {'lots': -0.02, 'notional': 2000, 'weight': 0.06, 'yf_ticker': 'EURJPY=X'},
    'GBPJPY': {'lots': 0.02, 'notional': 2300, 'weight': 0.07, 'yf_ticker': 'GBPJPY=X'}
}

# Default pairs trading (same as before)
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

def update_portfolio_from_dataframe(df):
    """Update the portfolio configuration from a dataframe"""
    try:
        # Create new portfolio configuration
        new_portfolio_config = {}
        
        # Map assets to Yahoo Finance tickers
        for _, row in df.iterrows():
            asset = row['Asset']
            
            # Determine Yahoo Finance ticker
            yf_ticker = asset
            if asset in ['XAUUSD', 'XAGUSD']:
                yf_ticker = 'GC=F' if asset == 'XAUUSD' else 'SI=F'
            elif asset in ['AUDJPY', 'EURCHF', 'USDJPY', 'EURUSD', 'GBPUSD']:
                yf_ticker = asset + '=X'
            elif asset == 'GDAXI':
                yf_ticker = '^GDAXI'
            elif asset == 'XTIUSD':
                yf_ticker = 'CL=F'
            
            new_portfolio_config[asset] = {
                'lots': float(row['Size']),
                'notional': float(row['Notional_Value']),
                'weight': float(row['Weight']),
                'yf_ticker': yf_ticker
            }
        
        # Update session state
        st.session_state['portfolio_config'] = new_portfolio_config
        st.session_state['total_capital'] = sum(config['notional'] for config in new_portfolio_config.values())
        
        # Clear existing analysis to force recalculation
        for key in ['prices', 'returns', 'positions', 'lot_sizes']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Portfolio updated successfully! Click 'Analyze Portfolio' to recalculate with new weights.")
        st.balloons()
        
        # Show quick summary
        st.info(f"""
        **Updated Portfolio Summary:**
        - Total Assets: {len(new_portfolio_config)}
        - Total Capital: ${st.session_state['total_capital']:,.0f}
        - Long Positions: {sum(1 for c in new_portfolio_config.values() if c['lots'] > 0)}
        - Short Positions: {sum(1 for c in new_portfolio_config.values() if c['lots'] < 0)}
        """)
        
    except Exception as e:
        st.error(f"Error updating portfolio: {str(e)}")

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

def calculate_ou_parameters(spread):
    """Calculate Ornstein-Uhlenbeck process parameters for pairs trading"""
    from statsmodels.tsa.stattools import adfuller
    
    # Remove NaN values
    spread_clean = spread.dropna()
    
    # Check if we have enough data
    if len(spread_clean) < 20:
        return {
            'theta': 0,
            'mu': spread_clean.mean() if len(spread_clean) > 0 else 0,
            'sigma': spread_clean.std() if len(spread_clean) > 0 else 0,
            'half_life': np.inf,
            'current_spread': spread_clean.iloc[-1] if len(spread_clean) > 0 else 0,
            'spread_std': spread_clean.std() if len(spread_clean) > 0 else 0,
            'z_score': 0,
            'adf_statistic': 0,
            'adf_pvalue': 1,
            'is_stationary': False
        }
    
    # Check if spread is constant (no variation)
    if spread_clean.std() == 0 or spread_clean.max() == spread_clean.min():
        return {
            'theta': 0,
            'mu': spread_clean.mean(),
            'sigma': 0,
            'half_life': np.inf,
            'current_spread': spread_clean.iloc[-1],
            'spread_std': 0,
            'z_score': 0,
            'adf_statistic': 0,
            'adf_pvalue': 1,
            'is_stationary': False
        }
    
    # Check stationarity
    try:
        adf_result = adfuller(spread_clean)
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
    except Exception as e:
        # If ADF test fails for any reason, set default values
        adf_statistic = 0
        adf_pvalue = 1
    
    # Proper OU parameter estimation
    # The discrete OU process: X_{t+1} = a + b * X_t + epsilon
    # Where: a = mu * theta * dt, b = 1 - theta * dt
    
    spread_values = spread_clean.values
    spread_lag = spread_values[:-1]
    spread_current = spread_values[1:]
    
    # Check if we have enough variation for regression
    if len(spread_lag) < 2 or np.std(spread_lag) == 0:
        return {
            'theta': 0,
            'mu': spread_clean.mean(),
            'sigma': spread_clean.std(),
            'half_life': np.inf,
            'current_spread': spread_clean.iloc[-1],
            'spread_std': spread_clean.std(),
            'z_score': 0,
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'is_stationary': False
        }
    
    # OLS regression
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    y = spread_current
    
    # Solve normal equations: beta = (X'X)^{-1}X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = beta[0], beta[1]
    except:
        # If regression fails, return default values
        return {
            'theta': 0,
            'mu': spread_clean.mean(),
            'sigma': spread_clean.std(),
            'half_life': np.inf,
            'current_spread': spread_clean.iloc[-1],
            'spread_std': spread_clean.std(),
            'z_score': 0,
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'is_stationary': False
        }
    
    # Extract OU parameters (assuming daily data, dt = 1/252)
    dt = 1/252
    
    # Ensure b is in valid range for log
    if b <= 0 or b >= 1:
        theta = 0
    else:
        theta = -np.log(b) / dt  # Mean reversion speed (annualized)
    
    mu = a / (1 - b) if (b != 1 and abs(1-b) > 1e-10) else spread_clean.mean()  # Long-term mean
    
    # Calculate residual standard deviation
    residuals = y - (a + b * spread_lag)
    sigma = np.std(residuals) / np.sqrt(dt) if np.std(residuals) > 0 else 0  # Annualized volatility
    
    # Half-life of mean reversion
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    # Current spread statistics
    current_spread = spread_clean.iloc[-1]
    spread_std = spread_clean.std()
    z_score = (current_spread - mu) / spread_std if spread_std > 0 else 0
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'current_spread': current_spread,
        'spread_std': spread_std,
        'z_score': z_score,
        'adf_statistic': adf_statistic,
        'adf_pvalue': adf_pvalue,
        'is_stationary': adf_pvalue < 0.05
    }

def generate_trading_signals(prices, returns, portfolio_config, pairs_config=None):
    """Generate trading signals for all assets"""
    signals = {}
    
    for asset in prices.columns:
        current_price = prices[asset].iloc[-1]
        current_lots = portfolio_config.get(asset, {}).get('lots', 0)
        
        # Check if this is part of a pair
        is_pair_asset = False
        if pairs_config:
            for pair in pairs_config:
                if asset in [pair['long'], pair['short']]:
                    is_pair_asset = True
                    pair_info = pair
                    break
        
        if is_pair_asset and pairs_config:
            # Pairs trading signal using OU process
            long_asset = pair_info['long']
            short_asset = pair_info['short']
            
            if long_asset in prices.columns and short_asset in prices.columns:
                # Calculate spread
                spread = np.log(prices[long_asset] / prices[short_asset])
                
                # OU parameters
                ou_params = calculate_ou_parameters(spread)
                
                current_spread = ou_params['current_spread']
                z_score = ou_params['z_score']
                
                # Trading signals based on z-score
                if z_score > 2:
                    if asset == long_asset:
                        signal = 'SELL LONG (Spread too high)'
                        action = 'sell'
                    else:
                        signal = 'ADD SHORT (Spread too high)'
                        action = 'short'
                elif z_score < -2:
                    if asset == long_asset:
                        signal = 'ADD LONG (Spread too low)'
                        action = 'buy'
                    else:
                        signal = 'COVER SHORT (Spread too low)'
                        action = 'cover'
                elif abs(z_score) < 0.5:
                    signal = 'CLOSE PAIR (Near mean)'
                    action = 'close'
                else:
                    signal = 'HOLD PAIR'
                    action = 'hold'
                
                signals[asset] = {
                    'signal': signal,
                    'action': action,
                    'z_score': ou_params['z_score'],
                    'spread': ou_params['current_spread'],
                    'ou_mean': ou_params['mu'],
                    'half_life': ou_params['half_life'],
                    'is_stationary': ou_params['is_stationary'],
                    'pair_type': 'long' if asset == long_asset else 'short'
                }
        else:
            # Regular asset signals using multiple indicators
            # 1. EMA deviation
            ema_50 = prices[asset].ewm(span=50, adjust=False).mean()
            ema_200 = prices[asset].ewm(span=200, adjust=False).mean()
            
            price_to_ema50 = (current_price - ema_50.iloc[-1]) / ema_50.iloc[-1]
            ema50_std = (prices[asset] / ema_50 - 1).std()
            z_score_ema = price_to_ema50 / ema50_std if ema50_std != 0 else 0
            
            # 2. RSI
            delta = prices[asset].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # 3. Momentum
            momentum_20 = (prices[asset].iloc[-1] / prices[asset].iloc[-20] - 1) * 100 if len(prices) > 20 else 0
            
            # 4. Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = prices[asset].rolling(window=bb_period).mean()
            std = prices[asset].rolling(window=bb_period).std()
            upper_band = sma + (bb_std * std)
            lower_band = sma - (bb_std * std)
            bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            # Combined signal
            signals_count = 0
            bull_signals = 0
            
            # EMA signal (stronger weight for extreme deviations)
            if z_score_ema < -3:
                bull_signals += 2
            elif z_score_ema < -2:
                bull_signals += 1
            elif z_score_ema > 3:
                bull_signals -= 2
            elif z_score_ema > 2:
                bull_signals -= 1
            signals_count += 2
            
            # RSI signal
            if rsi < 30:
                bull_signals += 1
            elif rsi > 70:
                bull_signals -= 1
            signals_count += 1
            
            # Bollinger Bands signal
            if bb_position < 0.2:
                bull_signals += 1
            elif bb_position > 0.8:
                bull_signals -= 1
            signals_count += 1
            
            # Trend signal
            if ema_50.iloc[-1] > ema_200.iloc[-1] and current_price > ema_50.iloc[-1]:
                bull_signals += 0.5
            elif ema_50.iloc[-1] < ema_200.iloc[-1] and current_price < ema_50.iloc[-1]:
                bull_signals -= 0.5
            
            # Generate final signal
            signal_strength = bull_signals / signals_count
            
            if current_lots > 0:  # Currently long
                if signal_strength < -0.5:
                    signal = 'SELL (Exit Long)'
                    action = 'sell'
                elif signal_strength < -0.25:
                    signal = 'REDUCE POSITION'
                    action = 'reduce'
                elif signal_strength > 0.5:
                    signal = 'ADD TO POSITION'
                    action = 'add'
                else:
                    signal = 'HOLD'
                    action = 'hold'
            elif current_lots < 0:  # Currently short
                if signal_strength > 0.5:
                    signal = 'COVER (Exit Short)'
                    action = 'cover'
                elif signal_strength > 0.25:
                    signal = 'REDUCE SHORT'
                    action = 'reduce'
                elif signal_strength < -0.5:
                    signal = 'ADD TO SHORT'
                    action = 'add'
                else:
                    signal = 'HOLD'
                    action = 'hold'
            else:  # No position
                if signal_strength > 0.5:
                    signal = 'BUY'
                    action = 'buy'
                elif signal_strength < -0.5:
                    signal = 'SHORT'
                    action = 'short'
                else:
                    signal = 'WAIT'
                    action = 'wait'
            
            signals[asset] = {
                'signal': signal,
                'action': action,
                'z_score_ema': z_score_ema,
                'rsi': rsi,
                'momentum_20': momentum_20,
                'bb_position': bb_position,
                'signal_strength': signal_strength,
                'ema_50': ema_50.iloc[-1],
                'ema_200': ema_200.iloc[-1] if len(ema_200) > 0 else None
            }
    
    return signals

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

def monte_carlo_portfolio_optimization(returns, current_positions, total_capital, n_simulations=10000):
    """Run Monte Carlo simulations to find optimal portfolio adjustments"""
    np.random.seed(42)
    
    # Current weights
    current_weights = current_positions.iloc[-1] / total_capital
    
    # Historical statistics
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Generate random portfolio weights
    results = []
    
    for _ in range(n_simulations):
        # Random weights (allowing negative for shorts)
        weights = np.random.randn(len(returns.columns))
        weights = weights / np.sum(np.abs(weights))  # Normalize to sum of abs = 1
        
        # Constrain to reasonable leverage (gross exposure < 2)
        if np.sum(np.abs(weights)) > 2:
            weights = weights / np.sum(np.abs(weights)) * 2
        
        # Calculate metrics
        portfolio_return = weights @ mean_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Risk contribution
        marginal_contrib = cov_matrix @ weights
        contrib = weights * marginal_contrib
        risk_contrib = contrib / portfolio_vol if portfolio_vol > 0 else contrib
        max_risk_contrib = np.max(np.abs(risk_contrib))
        
        results.append({
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe': sharpe,
            'max_risk_contrib': max_risk_contrib
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal portfolios
    # 1. Maximum Sharpe
    max_sharpe_idx = results_df['sharpe'].idxmax()
    max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
    
    # 2. Minimum risk concentration (most diversified)
    min_concentration_idx = results_df['max_risk_contrib'].idxmin()
    min_concentration_portfolio = results_df.iloc[min_concentration_idx]
    
    # 3. Target volatility (match current)
    current_vol = np.sqrt(current_weights @ cov_matrix @ current_weights)
    vol_diff = np.abs(results_df['volatility'] - current_vol)
    target_vol_idx = vol_diff.idxmin()
    target_vol_portfolio = results_df.iloc[target_vol_idx]
    
    return {
        'current_weights': current_weights,
        'max_sharpe': max_sharpe_portfolio,
        'min_concentration': min_concentration_portfolio,
        'target_vol': target_vol_portfolio,
        'current_vol': current_vol
    }

def calculate_market_regime(prices, returns, vix_threshold=25, lookback_days=20):
    """Detect market regime and stress indicators"""
    
    # Calculate rolling volatility for the portfolio
    portfolio_returns = returns.mean(axis=1)  # Simple average of returns
    rolling_vol = portfolio_returns.rolling(window=lookback_days).std() * np.sqrt(252)
    current_vol = rolling_vol.iloc[-1]
    vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100
    
    # Calculate recent drawdowns
    recent_returns = returns.iloc[-lookback_days:]
    cumulative_returns = (1 + recent_returns).prod() - 1
    avg_cumulative_return = cumulative_returns.mean()
    
    # Fetch VIX data
    try:
        end_date = prices.index[-1] + pd.Timedelta(days=1)
        start_date = prices.index[-30]
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if not vix.empty and 'Close' in vix.columns:
            current_vix = float(vix['Close'].iloc[-1])
            if len(vix) > 10:
                vix_ma = float(vix['Close'].rolling(10).mean().iloc[-1])
            else:
                vix_ma = current_vix
        else:
            current_vix = 20.0
            vix_ma = 20.0
    except:
        current_vix = 20.0  # Default if VIX fetch fails
        vix_ma = 20.0
    
    # Z-score for unusual moves (portfolio level)
    returns_mean = portfolio_returns.mean()
    returns_std = portfolio_returns.std()
    recent_avg_return = recent_returns.mean(axis=1).mean()
    portfolio_z_score = (recent_avg_return - returns_mean) / returns_std if returns_std > 0 else 0
    
    # Detect stress conditions - ensure all are boolean
    stress_indicators = {
        'high_vix': current_vix > vix_threshold,
        'vix_spike': current_vix > vix_ma * 1.2,  # 20% above average
        'high_volatility': vol_percentile > 80,
        'extreme_drawdown': avg_cumulative_return < -0.05,  # 5% drawdown
        'z_score_breach': abs(portfolio_z_score) > 2  # Portfolio level z-score
    }
    
    # Determine regime
    stress_count = sum(stress_indicators.values())
    if stress_count >= 3:
        regime = 'CRISIS'
        risk_multiplier = 0.5  # Cut risk by 50%
    elif stress_count >= 2:
        regime = 'STRESS'
        risk_multiplier = 0.7  # Cut risk by 30%
    elif stress_count >= 1:
        regime = 'CAUTION'
        risk_multiplier = 0.85  # Cut risk by 15%
    else:
        regime = 'NORMAL'
        risk_multiplier = 1.0
    
    return {
        'regime': regime,
        'risk_multiplier': risk_multiplier,
        'current_vix': float(current_vix),
        'vol_percentile': float(vol_percentile),
        'recent_drawdown': float(avg_cumulative_return * 100),
        'stress_indicators': stress_indicators,
        'portfolio_z_score': float(portfolio_z_score)
    }

def generate_adaptive_signals(signals, market_regime, portfolio_config):
    """Adjust trading signals based on market regime"""
    
    adaptive_signals = {}
    regime = market_regime['regime']
    risk_mult = market_regime['risk_multiplier']
    
    # Define risk-on and risk-off assets
    risk_on_assets = ['SPY', 'QQQ', 'IWM', 'GDAXI']
    risk_off_assets = ['TLT', 'XAUUSD', 'USDJPY', 'EURCHF']  # Bonds, Gold, Safe currencies
    
    for asset, signal_info in signals.items():
        adaptive_signal = signal_info.copy()
        current_lots = portfolio_config.get(asset, {}).get('lots', 0)
        
        if regime in ['CRISIS', 'STRESS']:
            # Crisis mode adjustments
            if asset in risk_on_assets:
                if current_lots > 0:  # Long risk-on positions
                    if regime == 'CRISIS':
                        adaptive_signal['signal'] = 'REDUCE BY 50% (Crisis Mode)'
                        adaptive_signal['action'] = 'reduce_crisis'
                    else:
                        adaptive_signal['signal'] = 'REDUCE BY 30% (Stress Mode)'
                        adaptive_signal['action'] = 'reduce_stress'
                elif current_lots < 0:  # Short risk-on positions
                    adaptive_signal['signal'] = 'HOLD/ADD SHORT (Risk-Off Mode)'
                    adaptive_signal['action'] = 'hold'
            
            elif asset in risk_off_assets:
                if current_lots > 0:  # Long risk-off positions
                    adaptive_signal['signal'] = 'ADD POSITION (Safe Haven)'
                    adaptive_signal['action'] = 'add_defensive'
                elif current_lots < 0:  # Short risk-off positions
                    adaptive_signal['signal'] = 'COVER SHORT (Risk-Off Mode)'
                    adaptive_signal['action'] = 'cover_defensive'
        
        elif regime == 'CAUTION':
            # Caution mode - moderate adjustments
            if signal_info['action'] in ['buy', 'add']:
                adaptive_signal['signal'] += ' (Reduced Size - Caution)'
                adaptive_signal['size_adjustment'] = 0.5  # Half position size
            elif signal_info['action'] in ['sell', 'short']:
                adaptive_signal['signal'] += ' (Caution Mode)'
        
        # Add regime info to signal
        adaptive_signal['market_regime'] = regime
        adaptive_signal['risk_multiplier'] = risk_mult
        adaptive_signals[asset] = adaptive_signal
    
    return adaptive_signals

def calculate_dynamic_position_sizing(portfolio_config, market_regime, total_capital):
    """Calculate adjusted position sizes based on market regime"""
    
    risk_mult = market_regime['risk_multiplier']
    regime = market_regime['regime']
    
    adjusted_positions = {}
    for asset, config in portfolio_config.items():
        original_notional = config['notional']
        
        # Adjust based on regime
        if regime in ['CRISIS', 'STRESS']:
            # Define asset categories
            if asset in ['SPY', 'QQQ', 'IWM', 'GDAXI']:  # Risk-on
                adjusted_notional = original_notional * risk_mult
            elif asset in ['TLT', 'XAUUSD']:  # Safe havens
                adjusted_notional = original_notional * (2 - risk_mult)  # Increase safe havens
            else:  # Neutral assets
                adjusted_notional = original_notional * ((1 + risk_mult) / 2)
        else:
            adjusted_notional = original_notional
        
        adjusted_positions[asset] = {
            'original': original_notional,
            'adjusted': adjusted_notional,
            'change_pct': (adjusted_notional / original_notional - 1) * 100
        }
    
    return adjusted_positions

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")

# Portfolio selection
portfolio_choice = st.sidebar.selectbox(
    "Select Portfolio",
    ["Vantage Portfolio", "Custom Portfolio"],
    key="portfolio_selector"
)

# Clear MC results if portfolio changed
if 'last_portfolio_choice' in st.session_state:
    if st.session_state['last_portfolio_choice'] != portfolio_choice:
        if 'mc_results' in st.session_state:
            del st.session_state['mc_results']
        if 'mc_completed' in st.session_state:
            del st.session_state['mc_completed']

st.session_state['last_portfolio_choice'] = portfolio_choice


col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Portfolio setup based on choice
            
if portfolio_choice == "Vantage Portfolio":
    portfolio_config = VANTAGE_PORTFOLIO.copy()
    st.sidebar.success("Using Vantage portfolio configuration")
    
    # Show Vantage portfolio
    with st.sidebar.expander("View Vantage Portfolio"):
        for asset, config in VANTAGE_PORTFOLIO.items():
            st.write(f"**{asset}**: {config['lots']} lots @ ${config['notional']:,.0f}")
else:  # Custom Portfolio
    st.sidebar.subheader("Custom Portfolio Setup")
    st.sidebar.info("Enter assets with lots and notional values")
    
    # Custom portfolio input
    num_assets = st.sidebar.number_input("Number of assets", 1, 20, 5)
    portfolio_config = {}
    
    # First, collect all assets without calculating weights
    temp_assets = []
    for i in range(num_assets):
        st.sidebar.write(f"Asset {i+1}")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            asset = st.text_input(f"Ticker", key=f"asset_{i}")
        with col2:
            lots = st.number_input(f"Lots", value=0.0, step=0.01, key=f"lots_{i}")
        with col3:
            notional = st.number_input(f"Notional $", value=10000, step=100, key=f"notional_{i}")
        
        if asset:
            # Map to Yahoo Finance ticker
            yf_ticker = asset
            if asset in ['XAUUSD', 'XAGUSD']:
                yf_ticker = 'GC=F' if asset == 'XAUUSD' else 'SI=F'
            elif asset in ['AUDJPY', 'EURCHF', 'USDJPY', 'EURUSD', 'GBPUSD']:
                yf_ticker = asset + '=X'
            elif asset == 'GDAXI':
                yf_ticker = '^GDAXI'
            elif asset == 'XTIUSD':
                yf_ticker = 'CL=F'
            
            temp_assets.append({
                'asset': asset,
                'lots': lots,
                'notional': notional,
                'yf_ticker': yf_ticker
            })
    
    # Calculate total capital and weights
    if temp_assets:
        total_notional = sum(item['notional'] for item in temp_assets)
        for item in temp_assets:
            portfolio_config[item['asset']] = {
                'lots': item['lots'],
                'notional': item['notional'],
                'weight': item['notional'] / total_notional if total_notional > 0 else 0,
                'yf_ticker': item['yf_ticker']
            }
            
# Calculate total capital from portfolio
total_capital = sum(config['notional'] for config in portfolio_config.values())
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
    num_pairs = st.sidebar.number_input("Number of pairs", 0, 5, 0)
    pairs_config = []
    assets_list = list(portfolio_config.keys())
    
    for i in range(num_pairs):
        st.sidebar.write(f"Pair {i+1}")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            long_asset = st.selectbox(f"Long asset {i+1}", assets_list, key=f"long_{i}")
        with col2:
            short_asset = st.selectbox(f"Short asset {i+1}", assets_list, key=f"short_{i}")
        pairs_config.append({'long': long_asset, 'short': short_asset})

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
                    'Current Price': f"${prices[asset].iloc[-1]:.2f}",
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
    
        # Risk metrics
        st.header("⚠️ Risk Metrics")
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
    st.info("👈 Please configure your portfolio in the sidebar and click 'Analyze Portfolio' to begin.")
    
    st.markdown("""
    ### 📋 Instructions:
    
    1. **Use Default Portfolio**: Toggle to use the pre-configured portfolio or create custom
    2. **Configure Date Range**: Select the analysis period
    3. **For Custom Portfolio**: 
        - Enter asset tickers
        - Set lot sizes (negative for shorts)
        - Set notional values per position
    4. **Configure Pairs**: Set up pairs trading relationships
    5. **Click 'Analyze Portfolio'** to generate analysis
    
    ### 📊 Darwin Portfolio Includes:
    - **Forex**: AUDJPY (Long), EURCHF (Short), USDJPY (Short)
    - **Indices**: DAX (Long), IWM/Russell 2000 (Short), QQQ/Nasdaq (Long), SPY/S&P500 (Long)
    - **Bonds**: TLT (Long)
    - **Commodities**: Gold (Long), Silver (Long), Oil (Long)

    ### 🎯 Vantage Portfolio Includes:
    - **Forex**: AUDJPY (Long), EURCHF (Short), USDJPY (Short)
    - **Indices**: DAX (Long), SPY/S&P500 (Long)
    - **Bonds**: TLT (Long)
    - **Commodities**: Silver (Long), Oil (Long)
    - Smaller position sizes with different risk profile
    
    ### 🔄 Default Pairs:
    - **XAUUSD/XAGUSD**: Gold vs Silver spread
    - **QQQ/IWM**: Tech vs Small Cap spread
    
    ### 📈 Features:
    - **Performance Metrics**: Sharpe, Sortino, Profit Factor, etc.
    - **Risk Analysis**: VaR, CVaR, Maximum Drawdown
    - **Position Sizing**: Handles lots and notional values
    - **Trading Signals**: 
        - EMA deviation with z-scores
        - RSI (oversold/overbought)
        - Bollinger Bands
        - Momentum indicators
        - Ornstein-Uhlenbeck process for pairs
    """)

# Footer
st.markdown("---")
st.markdown("*Note: This is for educational purposes. Always do your own research before making investment decisions.*")