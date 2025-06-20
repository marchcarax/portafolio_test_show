import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta
import warnings
from io import StringIO
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Analysis", 
        "🔄 Update Portfolio", 
        "🎯 Trading Signals", 
        "⚖️ Risk & Optimization",
        "📈 Market Regime"
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

    with tab2:  # Update Portfolio Tab
        st.header("🔄 Update Portfolio Weights & Sizes")
        
        # Create sub-tabs for different update methods
        update_tab1, update_tab2, update_tab3 = st.tabs(["Paste Data", "Manual Update", "File Upload"])
        
        with update_tab1:
            st.subheader("📋 Paste Portfolio Data")
            st.info("Paste your portfolio data in CSV format with columns: Asset, Size, Notional_Value, Weight")
            
            # Example format
            with st.expander("See example format"):
                st.code("""Asset,Size,Notional_Value,Weight
AUDJPY,0.01,561,0.05
EURCHF,-0.01,1000,0.09
GDAXI,0.1,2344,0.20
SPY,0.3,1552,0.13
XTIUSD,0.01,647,0.06
USDJPY,-0.02,1731,0.15
TLT,20,1917,0.17
XAGUSD,0.01,1807,0.16""")
            
            # Text area for pasting data
            paste_data = st.text_area(
                "Paste your data here (CSV format):",
                height=300,
                placeholder="""Asset,Size,Notional_Value,Weight
AUDJPY,0.2,12973,0.07
EURCHF,-0.05,5774,0.03
GDAXI,0.1,27032,0.15
..."""
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Parse Data", type="primary", key="parse_paste_data"):
                    if paste_data.strip():
                        try:
                            # Parse the pasted data
                            df = pd.read_csv(StringIO(paste_data))
                            
                            # Validate columns
                            required_cols = ['Asset', 'Size', 'Notional_Value', 'Weight']
                            missing_cols = [col for col in required_cols if col not in df.columns]
                            
                            if missing_cols:
                                st.error(f"Missing columns: {', '.join(missing_cols)}")
                            else:
                                # Clean and process data
                                df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
                                df['Notional_Value'] = pd.to_numeric(df['Notional_Value'], errors='coerce')
                                df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
                                
                                # Remove any rows with NaN values
                                df_clean = df.dropna()
                                
                                if len(df_clean) < len(df):
                                    st.warning(f"Removed {len(df) - len(df_clean)} rows with invalid data")
                                
                                # Store in session state
                                st.session_state['new_portfolio_df'] = df_clean
                                st.success(f"Successfully parsed {len(df_clean)} assets!")
                                
                        except Exception as e:
                            st.error(f"Error parsing data: {str(e)}")
                    else:
                        st.warning("Please paste some data first")
            
            # Display parsed data if available
            if 'new_portfolio_df' in st.session_state:
                st.subheader("📊 Parsed Portfolio Data")
                df_display = st.session_state['new_portfolio_df'].copy()
                
                # Format for display
                df_display['Notional_Value'] = df_display['Notional_Value'].apply(lambda x: f"${x:,.0f}")
                df_display['Weight'] = df_display['Weight'].apply(lambda x: f"{x*100:.1f}%")
                
                st.dataframe(df_display, use_container_width=True)
                
                # Update portfolio button
                if st.button("Update Portfolio with New Data", type="primary", key="update_from_paste"):
                    update_portfolio_from_dataframe(st.session_state['new_portfolio_df'])
        
        with update_tab2:
            st.subheader("✏️ Manual Portfolio Update")
            
            if 'portfolio_config' in st.session_state:
                current_portfolio = st.session_state['portfolio_config']
                
                # Create editable dataframe
                edit_data = []
                for asset, config in current_portfolio.items():
                    edit_data.append({
                        'Asset': asset,
                        'Size': config['lots'],
                        'Notional_Value': config['notional'],
                        'Weight': config['weight']
                    })
                
                edit_df = pd.DataFrame(edit_data)
                
                st.info("Edit the values below and click 'Apply Changes' to update the portfolio")
                
                # Use st.data_editor for inline editing
                edited_df = st.data_editor(
                    edit_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Asset": st.column_config.TextColumn("Asset", required=True),
                        "Size": st.column_config.NumberColumn("Size (Lots)", format="%.4f", required=True),
                        "Notional_Value": st.column_config.NumberColumn("Notional Value", format="$%.0f", required=True),
                        "Weight": st.column_config.NumberColumn("Weight", format="%.4f", min_value=0, max_value=1)
                    },
                    key="portfolio_editor"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Apply Changes", type="primary", key="apply_manual_changes"):
                        # Recalculate weights based on notional values
                        total_notional = edited_df['Notional_Value'].sum()
                        edited_df['Weight'] = edited_df['Notional_Value'] / total_notional
                        
                        st.session_state['new_portfolio_df'] = edited_df
                        update_portfolio_from_dataframe(edited_df)
                        
                with col2:
                    if st.button("Reset to Original", key="reset_manual"):
                        st.rerun()
                        
            else:
                st.info("Please analyze a portfolio first to enable manual editing")
        
        with update_tab3:
            st.subheader("📁 Upload Portfolio File")
            
            st.info("Upload a CSV file with columns: Asset, Size, Notional_Value, Weight")
            
            # Download template button
            template_data = pd.DataFrame({
                'Asset': ['AUDJPY', 'EURCHF', 'GDAXI', 'SPY', 'TLT'],
                'Size': [0.01, -0.01, 0.1, 0.3, 20],
                'Notional_Value': [561, 1000, 2344, 1552, 1917],
                'Weight': [0.05, 0.09, 0.20, 0.13, 0.17]
            })
            
            csv = template_data.to_csv(index=False)
            st.download_button(
                label="Download Template CSV",
                data=csv,
                file_name="portfolio_template.csv",
                mime="text/csv",
                key="download_template"
            )
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="File should contain columns: Asset, Size, Notional_Value, Weight",
                key="portfolio_upload"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate and clean
                    required_cols = ['Asset', 'Size', 'Notional_Value', 'Weight']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                    else:
                        # Process data
                        df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
                        df['Notional_Value'] = pd.to_numeric(df['Notional_Value'], errors='coerce')
                        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
                        df_clean = df.dropna()
                        
                        # Display
                        st.subheader("📊 Uploaded Portfolio Data")
                        df_display = df_clean.copy()
                        df_display['Notional_Value'] = df_display['Notional_Value'].apply(lambda x: f"${x:,.0f}")
                        df_display['Weight'] = df_display['Weight'].apply(lambda x: f"{x*100:.1f}%")
                        st.dataframe(df_display, use_container_width=True)
                        
                        if st.button("Update Portfolio with Uploaded Data", type="primary", key="update_from_upload"):
                            update_portfolio_from_dataframe(df_clean)
                            
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Show comparison if new portfolio is ready
        if 'new_portfolio_df' in st.session_state and 'portfolio_config' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Portfolio Comparison")
            
            new_df = st.session_state['new_portfolio_df']
            current_config = st.session_state['portfolio_config']
            
            # Create comparison dataframe
            comparison_data = []
            
            # Get all unique assets
            all_assets = set(new_df['Asset'].tolist()) | set(current_config.keys())
            
            for asset in sorted(all_assets):
                current_size = current_config.get(asset, {}).get('lots', 0)
                current_notional = current_config.get(asset, {}).get('notional', 0)
                
                new_row = new_df[new_df['Asset'] == asset]
                if not new_row.empty:
                    new_size = new_row.iloc[0]['Size']
                    new_notional = new_row.iloc[0]['Notional_Value']
                else:
                    new_size = 0
                    new_notional = 0
                
                size_change = new_size - current_size
                notional_change = new_notional - current_notional
                
                comparison_data.append({
                    'Asset': asset,
                    'Current Size': current_size,
                    'New Size': new_size,
                    'Size Change': size_change,
                    'Current Notional': current_notional,
                    'New Notional': new_notional,
                    'Notional Change': notional_change
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Format and style
            def highlight_changes(row):
                styles = [''] * len(row)
                if abs(row['Size Change']) > 0.01:
                    styles[3] = 'background-color: #FFE4B5'
                if abs(row['Notional Change']) > 100:
                    styles[6] = 'background-color: #FFE4B5'
                return styles
            
            styled_comparison = comparison_df.style.apply(highlight_changes, axis=1).format({
                'Current Size': '{:.4f}',
                'New Size': '{:.4f}',
                'Size Change': '{:+.4f}',
                'Current Notional': '${:,.0f}',
                'New Notional': '${:,.0f}',
                'Notional Change': '${:+,.0f}'
            })
            
            st.dataframe(styled_comparison, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                current_total = sum(config.get('notional', 0) for config in current_config.values())
                new_total = new_df['Notional_Value'].sum()
                st.metric("Total Portfolio Value", f"${new_total:,.0f}", f"${new_total - current_total:+,.0f}")
            with col2:
                assets_added = len([a for a in all_assets if a not in current_config and a in new_df['Asset'].values])
                assets_removed = len([a for a in all_assets if a in current_config and a not in new_df['Asset'].values])
                st.metric("Assets Added", assets_added)
                st.metric("Assets Removed", assets_removed)
            with col3:
                total_changes = len([1 for _, row in comparison_df.iterrows() if abs(row['Size Change']) > 0.01])
                st.metric("Total Changes", total_changes)

    with tab3:
        # Trading signals
        st.markdown("---")
        st.header("🎯 Trading Signals")
        
        # Market regime detection
        market_regime = calculate_market_regime(prices, returns)
        
        # Display market regime status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            regime_color = {
                'NORMAL': '🟢',
                'CAUTION': '🟡', 
                'STRESS': '🟠',
                'CRISIS': '🔴'
            }
            st.metric("Market Regime", 
                    f"{regime_color[market_regime['regime']]} {market_regime['regime']}")
        with col2:
            st.metric("VIX Level", f"{market_regime['current_vix']:.1f}")
        with col3:
            st.metric("Volatility Percentile", f"{market_regime['vol_percentile']:.0f}%")
        with col4:
            st.metric("Risk Multiplier", f"{market_regime['risk_multiplier']:.0%}")
        
        # Show stress indicators
        if market_regime['regime'] != 'NORMAL':
            with st.expander("⚠️ Active Stress Indicators", expanded=True):
                stress_msgs = []
                if market_regime['stress_indicators']['high_vix']:
                    stress_msgs.append(f"• VIX above threshold ({market_regime['current_vix']:.1f} > 25)")
                if market_regime['stress_indicators']['vix_spike']:
                    stress_msgs.append("• VIX spiked 20% above average")
                if market_regime['stress_indicators']['high_volatility']:
                    stress_msgs.append(f"• Volatility in top 20% historically")
                if market_regime['stress_indicators']['extreme_drawdown']:
                    stress_msgs.append(f"• Recent drawdown: {market_regime['recent_drawdown']:.1f}%")
                if market_regime['stress_indicators']['z_score_breach']:
                    stress_msgs.append(f"• Portfolio Z-score breach: {market_regime['portfolio_z_score']:.2f}")
                
                for msg in stress_msgs:
                    st.warning(msg)
        
        # Generate base and adaptive signals
        base_signals = generate_trading_signals(prices, returns, portfolio_config, pairs_config)
        adaptive_signals = generate_adaptive_signals(base_signals, market_regime, portfolio_config)
        
        # Show adaptive position sizing if not normal
        if market_regime['regime'] != 'NORMAL':
            st.subheader("🛡️ Defensive Position Adjustments")
            position_adjustments = calculate_dynamic_position_sizing(portfolio_config, market_regime, total_capital)
            
            adj_data = []
            for asset, adj in position_adjustments.items():
                if abs(adj['change_pct']) > 1:  # Only show significant changes
                    adj_data.append({
                        'Asset': asset,
                        'Original': f"${adj['original']:,.0f}",
                        'Adjusted': f"${adj['adjusted']:,.0f}",
                        'Change': f"{adj['change_pct']:+.0f}%"
                    })
            
            if adj_data:
                adj_df = pd.DataFrame(adj_data)
                st.dataframe(adj_df, use_container_width=True)
        
        # Create signal summary with adaptive signals
        signal_data = []
        for asset, signal_info in adaptive_signals.items():
            config = portfolio_config.get(asset, {})
            current_lots = config.get('lots', 0)
            current_price = prices[asset].iloc[-1]
            
            # Color code based on regime
            if market_regime['regime'] in ['CRISIS', 'STRESS']:
                if 'reduce' in signal_info.get('action', ''):
                    signal_display = f"🔴 {signal_info['signal']}"
                elif 'defensive' in signal_info.get('action', ''):
                    signal_display = f"🟢 {signal_info['signal']}"
                else:
                    signal_display = signal_info['signal']
            else:
                signal_display = signal_info['signal']
            
            signal_data.append({
                'Asset': asset,
                'Current Lots': f"{current_lots:.2f}",
                'Notional': f"${config.get('notional', 0):,.0f}",
                'Current Price': f"${current_price:.2f}",
                'Signal': signal_display,
                'Action': signal_info['action'].upper(),
                'Regime': market_regime['regime'],
                'Z-Score (EMA)': f"{signal_info.get('z_score_ema', 0):.2f}" if 'z_score_ema' in signal_info else 'N/A',
                'RSI': f"{signal_info.get('rsi', 0):.1f}" if 'rsi' in signal_info else 'N/A'
            })
        
        signal_df = pd.DataFrame(signal_data)
        
        # Color code actions
        def color_action(action):
            action_lower = action.lower()
            if any(x in action_lower for x in ['buy', 'add', 'cover']):
                return 'background-color: #90EE90'
            elif any(x in action_lower for x in ['sell', 'short', 'reduce']):
                return 'background-color: #FFB6C1'
            elif 'close' in action_lower:
                return 'background-color: #FFE4B5'
            elif any(x in action_lower for x in ['crisis', 'stress']):
                return 'background-color: #FF6B6B'
            elif 'defensive' in action_lower:
                return 'background-color: #4ECDC4'
            else:
                return ''
        
        styled_df = signal_df.style.applymap(color_action, subset=['Action'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed signal information
        st.subheader("📋 Detailed Signal Analysis")
        
        selected_asset = st.selectbox("Select asset for detailed analysis", prices.columns)
        
        if selected_asset and selected_asset in adaptive_signals:
            sig = adaptive_signals[selected_asset]
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{selected_asset} Signal Details:**")
                st.write(f"- **Action**: {sig['action'].upper()}")
                st.write(f"- **Signal**: {sig['signal']}")
                
                if 'z_score_ema' in sig:
                    st.write(f"- **EMA Z-Score**: {sig['z_score_ema']:.2f}")
                    st.write(f"- **RSI**: {sig['rsi']:.1f}")
                    st.write(f"- **20-Day Momentum**: {sig['momentum_20']:.1f}%")
                    st.write(f"- **BB Position**: {sig['bb_position']:.2f}")
            
            with col2:
                # Price chart with indicators
                fig = go.Figure()
                
                # Price
                fig.add_trace(go.Scatter(x=prices.index[-100:], y=prices[selected_asset].iloc[-100:],
                                        mode='lines', name='Price', line=dict(color='black', width=2)))
                
                # EMAs
                if 'ema_50' in sig and sig['ema_50'] is not None:
                    ema_50 = prices[selected_asset].ewm(span=50, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=prices.index[-100:], y=ema_50.iloc[-100:],
                                            mode='lines', name='EMA 50', line=dict(color='blue', width=1)))
                
                if 'ema_200' in sig and sig['ema_200'] is not None and len(prices) > 200:
                    ema_200 = prices[selected_asset].ewm(span=200, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=prices.index[-100:], y=ema_200.iloc[-100:],
                                            mode='lines', name='EMA 200', line=dict(color='red', width=1)))
                
                fig.update_layout(title=f'{selected_asset} Price Chart',
                                xaxis_title='Date', yaxis_title='Price',
                                height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Pairs trading details
        if pairs_config:
            st.subheader("🔄 Pairs Trading Analysis")
            for pair in pairs_config:
                if pair['long'] in prices.columns and pair['short'] in prices.columns:
                    spread = np.log(prices[pair['long']] / prices[pair['short']])
                    ou_params = calculate_ou_parameters(spread)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Plot normalized spread
                        normalized_spread = (spread - ou_params['mu']) / ou_params['spread_std']
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=normalized_spread.index, y=normalized_spread.values,
                                            mode='lines', name='Normalized Spread',
                                            line=dict(color='blue', width=2)))
                        
                        # Add threshold lines
                        fig.add_hline(y=0, line_dash="dash", 
                                    line_color="green", annotation_text="Mean")
                        fig.add_hline(y=2, line_dash="dash", 
                                    line_color="red", annotation_text="+2σ (Sell Long/Add Short)")
                        fig.add_hline(y=-2, line_dash="dash", 
                                    line_color="red", annotation_text="-2σ (Buy Long/Cover Short)")
                        fig.add_hline(y=1, line_dash="dot", line_color="orange", opacity=0.5)
                        fig.add_hline(y=-1, line_dash="dot", line_color="orange", opacity=0.5)
                        
                        fig.update_layout(
                            title=f'{pair["long"]}/{pair["short"]} Normalized Spread (Z-Score)',
                            xaxis_title='Date', 
                            yaxis_title='Z-Score',
                            height=400,
                            yaxis=dict(range=[-4, 4])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write(f"**{pair['long']}/{pair['short']} Analysis:**")
                        st.metric("Current Z-Score", f"{ou_params['z_score']:.2f}")
                        st.metric("Mean Reversion Speed (θ)", f"{ou_params['theta']:.3f}")
                        st.metric("Long-term Mean (μ)", f"{ou_params['mu']:.3f}")
                        st.metric("Half-life (days)", f"{ou_params['half_life']:.1f}")
                        st.metric("Stationary", "Yes ✓" if ou_params['is_stationary'] else "No ✗")
                        
                        # Clear trading recommendation
                        st.write("**Trading Action:**")
                        if ou_params['z_score'] > 2:
                            st.error(f"📉 SELL {pair['long']} / SHORT {pair['short']}")
                            st.write("Spread is too high, expect reversion")
                        elif ou_params['z_score'] < -2:
                            st.success(f"📈 BUY {pair['long']} / COVER {pair['short']}")
                            st.write("Spread is too low, expect reversion")
                        elif abs(ou_params['z_score']) < 0.5:
                            st.warning("⚖️ CLOSE POSITIONS")
                            st.write("Spread near mean")
                        else:
                            st.info("✋ HOLD POSITIONS")
                            st.write("Wait for better entry")
    
    with tab4:
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
        
        # Monte Carlo Optimization
        if st.button("Run Monte Carlo Optimization (10,000 simulations)", type="secondary", key="run_monte_carlo"):
            with st.spinner("Running simulations..."):
                mc_results = monte_carlo_portfolio_optimization(returns, positions, total_capital)
                
                # Store results in session state to prevent re-running
                st.session_state['mc_results'] = mc_results
                st.session_state['mc_completed'] = True
        
        # Display results if available
        if 'mc_completed' in st.session_state and st.session_state.get('mc_completed', False):
            mc_results = st.session_state.get('mc_results')
            
            if mc_results:
                st.subheader("📊 Optimization Results")
                
                # Display optimal portfolios
                optimal_portfolios = pd.DataFrame({
                    'Current': mc_results['current_weights'].values * 100,
                    'Max Sharpe': mc_results['max_sharpe']['weights'] * 100,
                    'Min Risk Concentration': mc_results['min_concentration']['weights'] * 100,
                    'Target Vol Match': mc_results['target_vol']['weights'] * 100
                }, index=returns.columns)
                
                st.write("**Suggested Portfolio Adjustments (%):**")
                styled_optimal = optimal_portfolios.style.format('{:.1f}').background_gradient(axis=1)
                st.dataframe(styled_optimal, use_container_width=True)
                
                # Metrics comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Volatility", f"{mc_results['current_vol']*100:.1f}%")
                    st.metric("Max Sharpe Vol", f"{mc_results['max_sharpe']['volatility']*100:.1f}%")
                with col2:
                    st.metric("Current Sharpe", "N/A")
                    st.metric("Max Sharpe Ratio", f"{mc_results['max_sharpe']['sharpe']:.3f}")
                with col3:
                    st.metric("Expected Return (Max Sharpe)", f"{mc_results['max_sharpe']['return']*100:.1f}%")
                    st.metric("Expected Return (Min Conc.)", f"{mc_results['min_concentration']['return']*100:.1f}%")
                
                # Specific recommendations
                st.subheader("💡 Recommendations")
                
                # Compare current vs optimal
                current_w = mc_results['current_weights'].values * 100
                optimal_w = mc_results['max_sharpe']['weights'] * 100
                
                adjustments = []
                for i, asset in enumerate(returns.columns):
                    diff = optimal_w[i] - current_w[i]
                    if abs(diff) > 5:  # Only show significant changes
                        if diff > 0:
                            adjustments.append(f"📈 **Increase {asset}** by {diff:.1f}%")
                        else:
                            adjustments.append(f"📉 **Decrease {asset}** by {abs(diff):.1f}%")
                
                if adjustments:
                    for adj in adjustments:
                        st.write(adj)
                else:
                    st.success("Portfolio is well-balanced!")
                
                # Add a clear button
                if st.button("Clear Results", key="clear_mc_results"):
                    del st.session_state['mc_results']
                    del st.session_state['mc_completed']
                    st.rerun()

    with tab5:  # Market Regime Tab
        st.header("📈 Market Regime Analysis")
        
        # Calculate current market regime
        market_regime = calculate_market_regime(prices, returns)
        
        # Display detailed regime analysis
        st.subheader("Current Market Conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            # Create a gauge chart for market stress
            stress_level = sum(market_regime['stress_indicators'].values())
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = stress_level,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Stress Level"},
                delta = {'reference': 1},
                gauge = {
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 5], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Stress Indicators Status:**")
            for indicator, status in market_regime['stress_indicators'].items():
                emoji = "🔴" if status else "🟢"
                indicator_name = indicator.replace('_', ' ').title()
                st.write(f"{emoji} **{indicator_name}**: {'Active' if status else 'Normal'}")
            
            st.markdown("---")
            st.write("**Regime Characteristics:**")
            st.write(f"- **Current Regime**: {market_regime['regime']}")
            st.write(f"- **Risk Multiplier**: {market_regime['risk_multiplier']:.0%}")
            st.write(f"- **VIX Level**: {market_regime['current_vix']:.1f}")
            st.write(f"- **Portfolio Z-Score**: {market_regime['portfolio_z_score']:.2f}")
        
        # Historical regime analysis
        st.subheader("Historical Market Regime Analysis")
        
        # Calculate historical regimes
        lookback_days = 20
        historical_regimes = []
        regime_dates = []
        
        # Calculate regime for each day in history
        for i in range(lookback_days, len(returns)):
            subset_prices = prices.iloc[:i+1]
            subset_returns = returns.iloc[:i+1]
            
            try:
                regime_data = calculate_market_regime(subset_prices, subset_returns, lookback_days=lookback_days)
                historical_regimes.append(regime_data['regime'])
                regime_dates.append(returns.index[i])
            except:
                continue
        
        if historical_regimes:
            # Create regime timeline
            regime_df = pd.DataFrame({
                'Date': regime_dates,
                'Regime': historical_regimes
            })
            
            # Map regimes to numeric values for visualization
            regime_map = {'NORMAL': 0, 'CAUTION': 1, 'STRESS': 2, 'CRISIS': 3}
            regime_df['Regime_Numeric'] = regime_df['Regime'].map(regime_map)
            
            # Plot regime timeline
            fig_regime = go.Figure()
            
            # Add colored backgrounds for different regimes
            colors = {'NORMAL': 'lightgreen', 'CAUTION': 'yellow', 'STRESS': 'orange', 'CRISIS': 'red'}
            
            for regime, color in colors.items():
                regime_data = regime_df[regime_df['Regime'] == regime]
                if not regime_data.empty:
                    fig_regime.add_trace(go.Scatter(
                        x=regime_data['Date'],
                        y=regime_data['Regime_Numeric'],
                        mode='markers',
                        name=regime,
                        marker=dict(color=color, size=10),
                        hovertemplate='%{x}<br>Regime: ' + regime + '<extra></extra>'
                    ))
            
            fig_regime.update_layout(
                title='Market Regime History',
                xaxis_title='Date',
                yaxis_title='Regime',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['NORMAL', 'CAUTION', 'STRESS', 'CRISIS']
                ),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_regime, use_container_width=True)
            
            # Regime statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Regime Distribution:**")
                regime_counts = regime_df['Regime'].value_counts()
                for regime in ['NORMAL', 'CAUTION', 'STRESS', 'CRISIS']:
                    count = regime_counts.get(regime, 0)
                    pct = count / len(regime_df) * 100
                    st.write(f"{regime}: {pct:.1f}%")
            
            with col2:
                st.write("**Average Duration (days):**")
                # Calculate average duration for each regime
                regime_changes = regime_df['Regime'] != regime_df['Regime'].shift()
                regime_groups = regime_changes.cumsum()
                
                for regime in ['NORMAL', 'CAUTION', 'STRESS', 'CRISIS']:
                    regime_periods = regime_df[regime_df['Regime'] == regime].groupby(regime_groups).size()
                    if len(regime_periods) > 0:
                        avg_duration = regime_periods.mean()
                        st.write(f"{regime}: {avg_duration:.1f}")
            
            with col3:
                st.write("**Recent Regime Changes:**")
                recent_changes = regime_df.tail(50)
                changes = recent_changes[recent_changes['Regime'] != recent_changes['Regime'].shift()]
                for _, change in changes.tail(5).iterrows():
                    st.write(f"{change['Date'].strftime('%Y-%m-%d')}: → {change['Regime']}")
        
        # Volatility analysis
        st.subheader("Volatility Analysis")
        
        # Calculate rolling volatility
        rolling_window = 20
        portfolio_returns = returns.mean(axis=1)
        rolling_vol = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
        
        # Plot volatility with regime overlay
        fig_vol = go.Figure()
        
        # Add volatility line
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            mode='lines',
            name='Portfolio Volatility (%)',
            line=dict(color='blue', width=2)
        ))
        
        # Add horizontal lines for volatility thresholds
        vol_percentiles = [25, 50, 75, 90]
        for percentile in vol_percentiles:
            threshold = np.percentile(rolling_vol.dropna(), percentile)
            fig_vol.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"P{percentile}: {threshold:.1f}%",
                annotation_position="right"
            )
        
        fig_vol.update_layout(
            title='Historical Portfolio Volatility',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility (%)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Risk metrics over time
        st.subheader("Risk Metrics Evolution")
        
        # Calculate rolling risk metrics
        window = 60  # 60 days rolling
        rolling_sharpe = []
        rolling_sortino = []
        rolling_max_dd = []
        dates = []
        
        for i in range(window, len(portfolio_returns)):
            window_returns = portfolio_returns.iloc[i-window:i]
            
            # Sharpe ratio
            sharpe = (window_returns.mean() * 252) / (window_returns.std() * np.sqrt(252))
            rolling_sharpe.append(sharpe)
            
            # Sortino ratio
            downside_returns = window_returns[window_returns < 0]
            sortino = (window_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
            rolling_sortino.append(sortino)
            
            # Max drawdown
            cum_returns = (1 + window_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            rolling_max_dd.append(drawdown.min() * 100)
            
            dates.append(portfolio_returns.index[i])
        
        # Plot risk metrics
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatter(
            x=dates,
            y=rolling_sharpe,
            mode='lines',
            name='Sharpe Ratio',
            line=dict(color='green', width=2)
        ))
        
        fig_risk.add_trace(go.Scatter(
            x=dates,
            y=rolling_sortino,
            mode='lines',
            name='Sortino Ratio',
            line=dict(color='blue', width=2)
        ))
        
        fig_risk.add_trace(go.Scatter(
            x=dates,
            y=rolling_max_dd,
            mode='lines',
            name='Max Drawdown (%)',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig_risk.update_layout(
            title='Rolling Risk Metrics (60-day window)',
            xaxis_title='Date',
            yaxis_title='Ratio',
            yaxis2=dict(
                title='Max Drawdown (%)',
                overlaying='y',
                side='right'
            ),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Recommendations based on regime
        st.subheader("💡 Regime-Based Recommendations")
        
        if market_regime['regime'] == 'CRISIS':
            st.error("""
            **CRISIS MODE - Immediate Actions Required:**
            - Reduce overall portfolio leverage by 50%
            - Exit or hedge high-beta positions
            - Increase allocation to safe-haven assets (Bonds, Gold)
            - Consider protective puts on equity positions
            - Monitor positions closely and set tight stop-losses
            """)
        elif market_regime['regime'] == 'STRESS':
            st.warning("""
            **STRESS MODE - Risk Reduction Advised:**
            - Reduce portfolio leverage by 30%
            - Trim positions in growth/momentum stocks
            - Increase cash reserves
            - Review and tighten stop-loss levels
            - Avoid new speculative positions
            """)
        elif market_regime['regime'] == 'CAUTION':
            st.info("""
            **CAUTION MODE - Heightened Awareness:**
            - Maintain current positions but avoid aggressive adds
            - Review portfolio risk metrics daily
            - Consider partial profit-taking on winners
            - Keep some dry powder for opportunities
            - Monitor regime indicators closely
            """)
        else:
            st.success("""
            **NORMAL MODE - Standard Operations:**
            - Follow standard position sizing rules
            - Implement signals as generated
            - Maintain target portfolio allocations
            - Look for new opportunities
            - Regular rebalancing schedule applies
            """)
    
    

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