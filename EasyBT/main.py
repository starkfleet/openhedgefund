import numpy as np
import pandas as pd
import tulipy as ti



# backtesting module
class bt:
    def backtester(indicator_list):
        commession = indicator_list['commission']
        qty = indicator_list['quantity']
        data = indicator_list['data']
        from_date = indicator_list['from_date']
        to_date = indicator_list['to_date']

        idx = 0
        data = pd.read_csv(
            data)[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
        data["datetime"] = pd.to_datetime(data["datetime"])

        buy_condition = indicator_list['buy_conditions'][0]
        sell_condition = indicator_list['buy_conditions'][0]

        # buy
        for indicator_frame in indicator_list['buy']:
            idx += 1
            sens = "buy"
            data = bt.saparator(indicator_frame,
                                from_date,
                                to_date,
                                data,
                                idx,
                                sens)

        # sell
        for indicator_frame in indicator_list['sell']:
            idx += 1
            sens = "sell"
            data = bt.saparator(indicator_frame,
                                from_date,
                                to_date,
                                data,
                                idx,
                                sens)

        data = bt.get_buy_sell_columns(data, buy_condition, sell_condition)[
            ['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'buy', 'sell']]
        
        return data

    def backtester_signal_csv(config):
        """Backtest pre-labeled signal CSVs (e.g., btc_data.csv).

        Expected CSV columns:
        - 'datetime' (parseable to datetime)
        - 'open', 'high', 'low', 'close', 'volume'
        - optional 'security_name' to filter by symbol
        - signal column (default: 'buy_sell_signal') with 1 for buy, -1 for sell, 0/NaN for none

        Config keys:
        - data: path to CSV file
        - from_date: inclusive start date (str or datetime-like), optional
        - to_date: inclusive end date (str or datetime-like), optional
        - symbol: optional symbol to match 'security_name'
        - signal_column: name of signal column (default 'buy_sell_signal')
        - buy_value: value representing buy in signal column (default 1)
        - sell_value: value representing sell in signal column (default -1)
        """

        csv_path = config.get('data')
        if csv_path is None:
            raise ValueError("config['data'] is required and should be a path to the CSV file")

        signal_column = config.get('signal_column', 'buy_sell_signal')
        buy_value = config.get('buy_value', 1)
        sell_value = config.get('sell_value', -1)

        df = pd.read_csv(csv_path)

        if 'datetime' not in df.columns:
            raise ValueError("CSV must include a 'datetime' column")

        # Normalize datetime
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Optional symbol filter when 'security_name' is present
        symbol = config.get('symbol')
        if symbol is not None and 'security_name' in df.columns:
            df = df[df['security_name'] == symbol]

        # Optional date filters
        from_date = config.get('from_date')
        to_date = config.get('to_date')
        if from_date is not None:
            start_dt = pd.to_datetime(from_date)
            df = df[df['datetime'] >= start_dt]
        if to_date is not None:
            end_dt = pd.to_datetime(to_date)
            df = df[df['datetime'] <= end_dt]

        if signal_column not in df.columns:
            raise ValueError(f"CSV must include a '{signal_column}' column")

        signal_series = df[signal_column].fillna(0)
        # Cast to int safely where possible
        try:
            signal_series = signal_series.astype(int)
        except Exception:
            # If non-integer values exist, map explicitly
            mapping = {str(buy_value): 1, str(sell_value): -1}
            signal_series = signal_series.astype(str).map(mapping).fillna(0).astype(int)

        # Derive buy/sell columns
        df['buy'] = (signal_series == buy_value).astype(int)
        df['sell'] = (signal_series == sell_value).astype(int)

        # Stable output column order where available
        preferred_columns = [
            'security_name', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell'
        ]
        existing_columns = [c for c in preferred_columns if c in df.columns]
        return df[existing_columns]

    def backtester_signal_csv_report(config):
        """Run a simple long-only backtest using pre-labeled signals in a CSV and return a detailed report.

        Trading rules:
        - Enter long when 'buy' == 1 and flat
        - Exit long when 'sell' == 1 and in position
        - One position at a time, all-in sizing by default
        - Equity is marked-to-market at each bar using the 'close' price

        Config extends backtester_signal_csv config with:
        - initial_cash: starting capital (default 100000)
        - commission_pct: commission per side as fraction of notional (default 0.001)
        - slippage_pct: slippage as fraction applied to trade price (default 0.0)
        - position_mode: 'all_in' (default), 'fixed_cash', or 'fixed_units'
        - fixed_cash_per_trade: cash allocated per entry when position_mode == 'fixed_cash'
        - fixed_units_per_trade: units per entry when position_mode == 'fixed_units'
        """

        # Prepare dataframe with signals
        df = bt.backtester_signal_csv(config)
        df = df.sort_values('datetime').reset_index(drop=True)

        initial_cash = float(config.get('initial_cash', 100000))
        commission_pct = float(config.get('commission_pct', 0.001))
        slippage_pct = float(config.get('slippage_pct', 0.0))
        position_mode = str(config.get('position_mode', 'all_in')).lower()
        fixed_cash_per_trade = float(config.get('fixed_cash_per_trade', initial_cash))
        fixed_units_per_trade = float(config.get('fixed_units_per_trade', 0.0))

        cash_balance = initial_cash
        position_units = 0.0
        entry_net_cost = 0.0
        entry_time = None
        entry_price_ex_slippage = None

        equity_points = []
        trade_records = []

        for _, row in df.iterrows():
            timestamp = row['datetime']
            price_close = float(row['close'])
            buy_signal = int(row['buy']) if 'buy' in row else 0
            sell_signal = int(row['sell']) if 'sell' in row else 0

            # Entry
            if buy_signal == 1 and position_units == 0.0:
                if position_mode == 'all_in':
                    trade_cash = cash_balance
                elif position_mode == 'fixed_cash':
                    trade_cash = min(fixed_cash_per_trade, cash_balance)
                elif position_mode == 'fixed_units':
                    trade_cash = None  # will be computed from units
                else:
                    trade_cash = cash_balance

                entry_trade_price = price_close * (1.0 + slippage_pct)

                if position_mode == 'fixed_units' and fixed_units_per_trade > 0.0:
                    units = fixed_units_per_trade
                    required_cash = units * entry_trade_price * (1.0 + commission_pct)
                    if required_cash > cash_balance:
                        units = 0.0
                        required_cash = 0.0
                else:
                    # Use all available trade_cash after fees
                    if entry_trade_price <= 0:
                        units = 0.0
                        required_cash = 0.0
                    else:
                        units = trade_cash / (entry_trade_price * (1.0 + commission_pct))
                        required_cash = units * entry_trade_price * (1.0 + commission_pct)

                if units > 0.0 and required_cash <= cash_balance:
                    cash_balance -= required_cash
                    position_units = units
                    entry_net_cost = required_cash
                    entry_time = timestamp
                    entry_price_ex_slippage = price_close

            # Exit
            if sell_signal == 1 and position_units > 0.0:
                exit_trade_price = price_close * (1.0 - slippage_pct)
                exit_net_proceeds = position_units * exit_trade_price * (1.0 - commission_pct)

                cash_balance += exit_net_proceeds

                pnl = exit_net_proceeds - entry_net_cost
                ret = pnl / entry_net_cost if entry_net_cost != 0 else 0.0
                duration = pd.to_datetime(timestamp) - pd.to_datetime(entry_time) if entry_time is not None else pd.Timedelta(0)

                trade_records.append({
                    'entry_datetime': entry_time,
                    'entry_price': entry_price_ex_slippage,
                    'units': position_units,
                    'exit_datetime': timestamp,
                    'exit_price': price_close,
                    'pnl': pnl,
                    'return': ret,
                    'duration': duration
                })

                # Flat the position
                position_units = 0.0
                entry_net_cost = 0.0
                entry_time = None
                entry_price_ex_slippage = None

            # Mark-to-market equity at close
            equity_value = cash_balance + position_units * price_close
            equity_points.append({'datetime': timestamp, 'equity': equity_value})

        equity_df = pd.DataFrame(equity_points)
        equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
        equity_df = equity_df.sort_values('datetime').reset_index(drop=True)
        equity_df['return'] = equity_df['equity'].pct_change().fillna(0.0)

        # Performance metrics
        start_equity = float(initial_cash)
        end_equity = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else start_equity
        total_return = (end_equity / start_equity) - 1.0 if start_equity != 0 else 0.0

        # Compute periods per year for annualization
        periods_per_year = 252.0
        if len(equity_df) > 1:
            dt = equity_df['datetime']
            median_days = np.median(np.diff(dt.values).astype('timedelta64[s]').astype(float)) / (24 * 3600)
            if median_days and median_days > 0:
                periods_per_year = 365.0 / float(median_days)

        returns = equity_df['return']
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret and std_ret > 0 else 0.0

        # Max drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        # CAGR
        if len(equity_df) >= 2:
            total_days = (equity_df['datetime'].iloc[-1] - equity_df['datetime'].iloc[0]).days
            years = total_days / 365.25 if total_days > 0 else 0.0
            cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if years and years > 0 else total_return
        else:
            cagr = total_return

        trades_df = pd.DataFrame(trade_records)
        num_trades = int(len(trades_df))
        wins = int((trades_df['pnl'] > 0).sum()) if num_trades > 0 else 0
        losses = int((trades_df['pnl'] < 0).sum()) if num_trades > 0 else 0
        win_rate = wins / num_trades if num_trades > 0 else 0.0
        avg_win = float(trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean()) if losses > 0 else 0.0
        gross_profit = float(trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()) if wins > 0 else 0.0
        gross_loss = float(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()) if losses > 0 else 0.0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf if gross_profit > 0 else 0.0

        summary = {
            'start_equity': start_equity,
            'end_equity': end_equity,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'sharpe': float(sharpe),
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_pct,
            'position_mode': position_mode
        }

        return {
            'summary': summary,
            'equity_curve': equity_df,
            'trades': trades_df,
            'signals': df
        }

    def saparator(ind_frame, from_date, to_date, data, idx, sens):

        indi_1 = ind_frame['indicator']
        values_1 = ind_frame['values']
        cal_on_1 = bt.__CalcOn__(ind_frame['on'])

        cond = bt.__ConditionFunction__(ind_frame['condition'])

        indi_2 = ind_frame['cross_indicator']
        values_2 = ind_frame['cross_values']
        cal_on_2 = bt.__CalcOn__(ind_frame['cross_on'])
        data = bt.slicer(from_date,
                         to_date,
                         data)

        data = bt.ta_calculate(indi_1,
                               values_1,
                               cal_on_1,
                               cond,
                               indi_2,
                               values_2,
                               cal_on_2,
                               data,
                               idx,
                               sens)

        return data

    def get_buy_sell_columns(df, buy_condition, sell_condition):
        col_array = df.columns.values
        buy_columns = [col for col in col_array if col.startswith("buy_")]
        buy_signal = df[buy_columns]
        buy_column = buy_signal.columns.values
        if len(buy_column) > 1:
            for i in range(len(buy_column) - 1):
                # Add parentheses to .to_numpy()
                m1 = buy_signal[buy_column[i]].to_numpy()
                # Add parentheses to .to_numpy()
                m2 = buy_signal[buy_column[i + 1]].to_numpy()
                con = buy_condition[i]
                if con == "or":
                    buy_signal[buy_column[i + 1]
                               ] = np.logical_or(m1, m2).astype(int)
                elif con == 'and':  # Use elif to avoid ambiguity
                    buy_signal[buy_column[i + 1]
                               ] = np.logical_and(m1, m2).astype(int)
        else:
            buy_signal = buy_signal.astype(int)
        df['buy'] = buy_signal.iloc[:, -1]

        sell_columns = [col for col in col_array if col.startswith("sell_")]
        sell_signal = df[sell_columns]
        sell_column = sell_signal.columns.values
        if len(sell_column) > 1:
            for i in range(len(sell_column) - 1):
                # Add parentheses to .to_numpy()
                m1 = sell_signal[sell_column[i]].to_numpy()
                # Add parentheses to .to_numpy()
                m2 = sell_signal[sell_column[i + 1]].to_numpy()
                con = sell_condition[i]
                if con == "or":
                    sell_signal[sell_column[i + 1]
                                ] = np.logical_or(m1, m2).astype(int)
                elif con == 'and':  # Use elif to avoid ambiguity
                    sell_signal[sell_column[i + 1]
                                ] = np.logical_and(m1, m2).astype(int)
        else:
            sell_signal = sell_signal.astype(int)
        df['sell'] = sell_signal.iloc[:, -1]
        return df

    def ta_calculate(ind, values, on, cond, ind_2, values_2, on_2, data, idx, snes):

        data = eval(f"ta.{ind}(data, on, values)")

        m1 = data[ind]

        data = eval(f"ta.{ind_2}(data, on_2, values_2)")

        m2 = data[ind_2]
        if cond == "crossover":
            signal = np.zeros_like(m1)
            for i in range(len(m1)):
                if m1[i] < m2[i] and m1[i - 1] >= m2[i - 1]:
                    if snes == "buy":
                        signal[i] = 1
                        column_name = "buy_"
                    else:
                        signal[i] = -1
                        column_name = "sell_"
                else:
                    if snes == "buy":
                        signal[i] = 0
                        column_name = "buy_"
                    else:
                        signal[i] = 0
                        column_name = "sell_"
            column_name = column_name+str(idx)
            data[column_name] = signal

        if cond == "crossunder":
            signal = np.zeros_like(m1)
            for i in range(len(m1)):
                if m1[i] > m2[i] and m1[i - 1] <= m2[i - 1]:
                    if snes == "buy":
                        signal[i] = 1
                        column_name = "buy_"
                    else:
                        signal[i] = -1
                        column_name = "sell_"
                else:
                    if snes == "buy":
                        signal[i] = 0
                        column_name = "buy_"
                    else:
                        signal[i] = 0
                        column_name = "sell_"
            column_name = column_name+str(idx)
            data[column_name] = signal

        if cond == "over":
            signal = np.zeros_like(m1)
            for i in range(len(m1)):
                if m1[i] > m2[i]:
                    if snes == "buy":
                        signal[i] = 1
                        column_name = "buy_"
                    else:
                        signal[i] = -1
                        column_name = "sell_"
                else:
                    if snes == "buy":
                        signal[i] = 0
                        column_name = "buy_"
                    else:
                        signal[i] = 0
                        column_name = "sell_"
            column_name = column_name+str(idx)
            data[column_name] = signal

        if cond == "under":
            signal = np.zeros_like(m1)
            for i in range(len(m1)):
                if m1[i] < m2[i]:
                    if snes == "buy":
                        signal[i] = 1
                        column_name = "buy_"
                    else:
                        signal[i] = -1
                        column_name = "sell_"
                else:
                    if snes == "buy":
                        signal[i] = 0
                        column_name = "buy_"
                    else:
                        signal[i] = 0
                        column_name = "sell_"
            column_name = column_name+str(idx)
            data[column_name] = signal
        return data

    def slicer(from_date, to_date, data):
        start_date = pd.to_datetime(from_date)
        end_date = pd.to_datetime(to_date)

        data = data[data["datetime"].between(start_date, end_date)]
        return data

    def __CalcOn__(num):
        switch = {
            0: "open",
            1: "high",
            2: "low",
            3: "close"
        }
        return switch.get(num, "Invalid on condition input")

    def __ConditionFunction__(num):
        switch = {
            0: "na",
            1: "crossover",
            2: "crossunder",
            3: "over",
            4: "under"
        }
        return switch.get(num, "Invalid condition input")
    
#technical library
class ta:
    def asin(data,on, values):
        d = data[on].values
        ans =  ti.asin(d)
        data["asin"]  = ans
        return data
    
    def atan(data,on, values):
        o = data[on].values
        ans =  ti.atan(o)
        data["atan"] = ans
        return data
    
    def atr(data, on, values):
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
    
        # Calculate Average True Range (ATR)
        atr = ti.atr(h, l, c, values[0])
        data['atr'] = atr
        return data
    
    def avgprice(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
    
        ans = ti.avgprice(o,h,l,c)
        data['avgprice'] = ans
        return data
    
    def ad(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.ad(h,l,c,v)
        data['ad'] = ans
        return data
    
    def adosc(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.adosc(h,l,c,v,values[0],values[1])
        data['adosc'] = ans
        return data
    
    def adx(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
        
        ans = ti.adx(h,l,c,values[0])
        data['adx'] = ans
        return data
    
    def adxr(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.adxr(h,l,c,values[0])
        data['adxr'] = ans
        return data
    
    def ao(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.ao(h,l)
        data['ao'] = ans
        return data
    
    def apo(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans  = ti.apo(o,values[0],values[1])
        data['apo'] = ans
        return data
    
    def aroon(data, on, values):
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        ans = ti.aroon(h,l,values[0])
        
        data['aroon'] = ans
        return data
    
    def aroonosc(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.aroonosc(h,l,values[0])
        data['aroonosc'] = ans 
        return data
    
    def asin(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.asin(o)
        data['asin'] = ans
        return data
    
    def atan(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.atan(o)
        data['atan'] = ans
        return data
    
    
    def upper_bband(data, on, values):
        o = np.array(data[on], dtype=float)
    
        upper_bband, mid_band, lower_band = ti.bbands(o,values[0], values[1])
    
        data["upper_bband"] = upper_bband
        return data
    
    def mid_bband(data, on, values):
        o = np.array(data[on], dtype=float)
    
        upper_bband, mid_band, lower_band = ti.bbands(o,values[0], values[1])
    
        data["mid_band"] = mid_band
        return data
    
    def lower_bband(data, on, values):
        o = np.array(data[on], dtype=float)
    
        upper_bband, mid_band, lower_band = ti.bbands(o,values[0], values[1])
    
        data["lower_band"] = lower_band
        return data
    
    def bop(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.bop(o,h,l,c)
        data['bop'] = ans
        return data
    
    def cci(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.cci(h,l,c,values[0])
        data['cci'] = ans
        return data
    
    def ceil(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.ceil(o)
        data['ceil'] = ans
        return data
    
    def cmo(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.cmo(o,values[0])
        data['cmo'] = ans
        return data
    
    def cos(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.cmo(o,values[0])
        data['cos'] = ans
        return data
    
    def cosh(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.cosh(o)
        data['cosh'] = ans
        return data
    
    def cvi(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.cvi(h,l,values[0])
        data['cvi'] = ans
        return data
    
    def decay(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.decay(o)
        data['decay'] = ans
        return data
    
    def dema(data, on, values):
        o = np.array(data[on], dtype=float)
    
    
        ans = ti.dema(o,values[0])
        data['dema'] = ans
        return data
    
    def di(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.di(h,l,c,values[0])
        data['di'] = ans
        return data
    
    def dm(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.dm(h,l,values[0])
        data['dm'] = ans
        return data
    
    def dpo(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.dpo(o, values[0])
        data['dpo'] = ans
        return data
    
    def dx(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.dx(h,l,c,values[0])
        data['dx'] = ans
        return data
    
    def edecay(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.edecay(o,values[0])
        data['edecay'] = ans
        return data
    
    def ema(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.ema(o,values[0])
        data['ema'] = ans
        return data
    
    def emv(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.emv(h,l,v)
        data['emv'] = ans
        return data
    
    def exp(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.exp(o)
        data['exp'] = ans
        return data
    
    def fisher(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.fisher(h,l,values[0])
        data['fisher'] = ans
        return data
    
    def floor(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.floor(o)
        data['floor'] = ans
        return data
    
    def fosc(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.fosc(o, values[0])
        data['fosc'] = ans
        return data
    
    def hma(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.hma(o, values[0])
        data['hma'] = ans
        return data
    
    def kama(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.kama(o, values[0])
        data['kama'] = ans
        return data
    
    def kvo(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)
    
        ans = ti.kvo(h,l,c,v,values[0],values[1])
        data['kvo'] = ans
        return data
    
    def lag(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.lag(o, values[0])
        data['lag'] = ans
        return data
    
    def linreg(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.linreg(o, values[0])
        data['linreg'] = ans
        return data
    
    def linregintercept(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.linreg(o, values[0])
        data['linregintercept'] = ans
        return data
    
    def linregslope(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.linregslope(o, values[0])
        data['linregslope'] = ans
        return data
    
    def ln(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.ln(o)
        data['ln'] = ans
        return data
    def log10(data, on, values):
        o = np.array(data[on], dtype=float)
    
        ans = ti.log10(o)
        data['log10'] = ans
        return data
    
    def macd(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        macd, macd_signal, macd_histogram = ti.macd(c,values[0],values[1],values[2])
        data['macd'] = macd
        return data
    
    def macd_signal(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        macd, macd_signal, macd_histogram = ti.macd(c,values[0],values[1],values[2])
        data['macd_signal'] = macd_signal
        return data
    
    def macd_histogram(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        macd, macd_signal, macd_histogram = ti.macd(c,values[0],values[1],values[2])
        data['macd_histogram'] = macd_histogram
        return data
    
    def marketfi(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.marketfi(h,l,v)
        data['marketfi'] = ans
        return data
    
    def mass(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.mass(h,l,values[0])
        data['mass'] = ans
        return data
    
    def max(data, on, values):
        o = np.array(data[on], dtype=float)

        ans=  ti.max(o, values[0])
        data['mass'] = ans
        return data
    
    def md(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.md(o, values[0])
        data['md'] = ans
        return data

    def medprice(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.medprice(h,l)
        data['medprice'] = ans
        return data
    
    def mfi(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.mfi(h,l,c,v,values[0])
        data['mfi'] = ans
        return data
    
    def min(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.min(o, values[0])
        data['min'] = ans
        return data

    def mom(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.mom(o, values[0])
        data['mom'] = ans
        return data
    
    def msw(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.msw(o, values[0])
        data['msw'] = ans
        return data
    
    def natr(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.natr(h,l,c,values[0])
        data['natr'] = ans
        return data
    
    def nvi(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.nvi(c,v)
        data['nvi'] = ans
        return data
    
    def obv(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.obv(c,v)
        data['obv'] = ans
        return data
    
    def ppo(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.ppo(o, values[0], values[1])
        data['ppo'] = ans
        return data
    
    def psar(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.psar(h,l,values[0], values[1])
        data['psar'] =  ans
        return data
    
    def pvi(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.pvi(c,v)
        data['pvi'] = ans
        return data
    
    def qstick(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.qstick(o,c,values[0])
        data['qstick'] = ans

        return data
    
    def roc(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.roc(o, values[0])
        data['roc'] = ans
        return data

    def rocr(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.rocr(o ,values[0])
        data['rocr'] = ans
        return data
    
    def round(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.round(o)
        data['round'] = ans
        return data
    
    def rsi(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.rsi(o, values[0])
        data['rsi'] = ans
        return data

    def sin(data,on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.sin(o)
        data['sin'] = ans
        return data
    
    def sinh(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.sinh(o)
        data['sinh'] = ans
        return data
    
    def sma(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.sma(o, values[0])
        data['sma'] = ans
        return data
    
    def sqrt(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.sqrt(o)
        data['sqrt'] = ans
        return data
    
    def stddev(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.stddev(o,values[0])
        data['stddev'] = ans
        return data
    
    def stderr(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.stderr(o,values[0])
        data['stderr'] = ans
        return data
    
    def stoch1(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans, bs = ti.stoch(h,l,c,values[0],values[1],values[2])
        data['stoch'] = ans
        return data
    
    def stoch2(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        bs, ans = ti.stoch(h,l,c,values[0],values[1],values[2])
        data['stoch'] = ans
        return data

    def stochrsi(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.stochrsi(o, values[0])
        data['stochrsi'] = ans
        return data
    
    def tan(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.tan(o)
        data['tan']  = ans
        return data
    
    def tanh(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.tanh(o)
        data['tanh']  = ans
        return data
    
    def tema(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.tema(o, values[0])
        data['tema'] = ans
        return data
    
    def todeg(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.todeg(o)
        data['todeg'] = ans
        return data
    
    def torad(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.torad(o)
        data['torad'] = ans
        return data
    
    def tr(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.tr(h,l,c)
        data['tr'] = ans
        return data
    
    def trima(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.trima(o)
        data['trima'] = ans
        return data
    
    def trix(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.trix(o,values[0])
        data['trix'] = ans
        return data
    
    def trunc(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.trunc(o)
        data['trunc'] = ans
        return data
    
    def tsf(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.tsf(o, values[0])
        data['tsf'] = ans
        return data
    
    def typprice(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.typprice(h,l,c)
        data['typprice'] = ans

        return data
    
    def ultosc(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.ultosc(h,l,c,values[0], values[1], values[2])
        data['ultosc'] = ans
        return data
    
    def var(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.var(o, values[0])
        data['var'] = ans
        return data
    
    def vhf(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.vhf(o, values[0])
        data['vhf']= ans
        return data
    
    def vidya(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.vidya(o, values[0])
        data['vidya'] = ans
        return data
    
    def volatility(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.volatility(o, values[0])
        data['volatility'] = ans
        return data
    
    def vosc(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.vosc(v, values[0], values[1])
        data['vosc'] = ans
        return data
    
    def vwma(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.vwma(c,v, values[0])
        data['vwma'] = ans
        return data
    
    def wad(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.wad(h,l,c)
        data['wad'] = ans
        return data
    
    def wcprice(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.wcprice(h,l,c)
        data['wcprice'] = ans
        return data
    
    def wilders(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.wilders(o,values[0])

        data['wilders'] = ans
        return data
    
    def willr(data, on, values):
        o = np.array(data.close, dtype=float)
        h = np.array(data.high, dtype=float)
        l = np.array(data.low, dtype=float)
        c = np.array(data.close, dtype=float)
        v = np.array(data.volume, dtype=float)

        ans = ti.willr(h,l,c,values[0])
        data['willr'] = ans
        return data
    
    def wma(data, on, values):
        o = np.array(data[on], dtype=float)

        ans = ti.wma(o, values[0])
        data['wma'] = ans
        return data
    
    def zlema(data, on, values):
        o = np.array(data[on], dtype=float)
        ans = ti.zlema(o, values[0])
        data['zlema'] = ans
        return data

