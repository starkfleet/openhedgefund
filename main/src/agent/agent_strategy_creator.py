from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import re
from textwrap import dedent
try:
    from .script_runner import run_sandboxed_python_script
except ImportError:  # fallback when running as a standalone script
    from script_runner import run_sandboxed_python_script
import pandas as pd
import io
from typing import Optional


class StrategyCreatorAgent:
    """Agent that generates, executes, and returns strategy results as a DataFrame."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: int = 2,
    ) -> None:
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _extract_python_code(script_string: str) -> str:
        m = re.search(r"```(?:python[^\n]*\n)?([\s\S]*?)```", script_string, re.IGNORECASE)
        if m:
            return dedent(m.group(1)).strip()
        lines = script_string.splitlines()
        start = next((i for i, ln in enumerate(lines)
                      if re.match(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", ln)), 0)
        return "\n".join(lines[start:]).strip()

    def _build_generation_messages(
        self,
        buy_condition: str,
        sell_condition: str,
        symbol: str,
        from_date: str,
        to_date: str,
        timeframe: str,
        api_url: str,
    ):
        return [
            (
                "system",
                "You are an expert Python programmer who specializes in financial algorithms. You only write clean, commented, and efficient code.",
            ),
            ("human", f"""
        ### **Objective**

    The script will fetch historical market data from a specified API endpoint, apply a user-defined trading strategy, generate buy/sell signals, and output the results in a strict CSV format to standard output.

    ### **Core Requirements**

    #### **Language & Libraries**

    * **Language:** Python 3.  
    * **Libraries:** You may only use the following external libraries: pandas, numpy, and requests. Do not use any other financial or plotting libraries.

    #### **Data Fetching**

    * The script must fetch data by making a POST request to the following URL: {api_url}/data.  
    * The request Content-Type header must be application/json.  
    * The request body must be a JSON object with the following structure:  
    {{
        "name": "{symbol}",
        "from_date": "{from_date}",
        "to_date": "{to_date}",
        "timeframe": "{timeframe}"
    }}

    * The name field must be one of: BTC, ETH, SOL.

        * The script must handle the API response, which is a JSON object in this format:  
    {{
        "success": true,
        "data": [
        {{
            "symbol": "BTCUSDT",
            "open": 44500.50,
            "high": 44750.25,
            "low": 44300.75,
            "close": 44650.00,
            "volume": 1250.75,
            "timestamp": "2024-01-01T00:00:00"
        }}
        ]
    }}

    * Include error handling for the API request (e.g., if the connection fails or returns a non-200 status code). If the fetch fails, the script should exit gracefully with an error message.

    #### **Strategy & Signal Generation Logic**

    1. Load the data array from the JSON response into a pandas DataFrame.  
    2. Convert the timestamp column to a pandas datetime object.  
    3. Create a new column named buy_sell_signal, initialized to 0.  
    4. Implement the following specific trading logic:  
    * **Buy Condition:** {buy_condition} 
    * **Sell Condition:** {sell_condition}  
    5. Based on the conditions, populate the buy_sell_signal column:  
    * 1 for a buy signal.  
    * -1 for a sell signal.  
    * 0 for no signal (hold).

    #### **Output Requirements (Strict)**

    1. Prepare a final DataFrame containing **ONLY** the following columns in **THIS EXACT ORDER**: ("security_name", "datetime", "open", "high", "low", "close", "volume", "buy_sell_signal").  
    2. The security_name column should be populated with the symbol value from the API response.  
    3. The datetime column should be the original timestamp.  
    4. The code will not have any type of comments just pure python script.
    5. The final step of the script **MUST** be to print the entire DataFrame to standard output in CSV format, without the index. Use this exact command:  
    print(final_df.to_csv(index=False))  

        
        """),
        ]

    def generate_strategy_creator_script(
        self,
        buy_condition: str = "SMA(10) > SMA(30)",
        sell_condition: str = "SMA(10) < SMA(30)",
        symbol: str = "BTC",
        from_date: str = "2024-01-01",
        to_date: str = "2025-01-01",
        timeframe: str = "1d",
        api_url: str = "https://97ne32z9yn5u.share.zrok.io",
    ) -> str:
        messages = self._build_generation_messages(
            buy_condition, sell_condition, symbol, from_date, to_date, timeframe, api_url
        )
        ai_msg = self._llm.invoke(messages)
        return ai_msg.content

    async def agenerate_strategy_creator_script(
        self,
        buy_condition: str = "SMA(10) > SMA(30)",
        sell_condition: str = "SMA(10) < SMA(30)",
        symbol: str = "BTC",
        from_date: str = "2024-01-01",
        to_date: str = "2025-01-01",
        timeframe: str = "1d",
        api_url: str = "https://97ne32z9yn5u.share.zrok.io",
    ) -> str:
        messages = self._build_generation_messages(
            buy_condition, sell_condition, symbol, from_date, to_date, timeframe, api_url
        )
        ai_msg = await self._llm.ainvoke(messages)
        return ai_msg.content

    def run_strategy_and_get_dataframe(
        self,
        buy_condition: str = "SMA(10) > SMA(30)",
        sell_condition: str = "SMA(10) < SMA(30)",
        symbol: str = "BTC",
        from_date: str = "2024-01-01",
        to_date: str = "2025-01-01",
        timeframe: str = "1d",
        api_url: str = "https://97ne32z9yn5u.share.zrok.io",
    ) -> pd.DataFrame:
        script_string = self.generate_strategy_creator_script(
            buy_condition, sell_condition, symbol, from_date, to_date, timeframe, api_url
        )
        code = self._extract_python_code(script_string)
        result, stderr, exit_code = run_sandboxed_python_script(code)

        if exit_code != 0:
            raise Exception(f"Script execution failed with exit code {exit_code}. Error: {stderr}")

        if isinstance(result, pd.DataFrame):
            return result

        if isinstance(result, str):
            try:
                df = pd.read_csv(io.StringIO(result))
                return df
            except Exception as e:
                raise Exception(f"Failed to parse CSV output: {e}. Raw output: {result}")

        raise Exception(f"Unexpected result type from runner: {type(result)}. STDERR: {stderr}")


def run_strategy_and_get_dataframe(
    buy_condition: str = "SMA(10) > SMA(30)",
    sell_condition: str = "SMA(10) < SMA(30)",
    symbol: str = "BTC",
    from_date: str = "2024-01-01",
    to_date: str = "2025-01-01",
    timeframe: str = "1d",
    api_url: str = "https://97ne32z9yn5u.share.zrok.io",
) -> pd.DataFrame:
    agent = StrategyCreatorAgent()
    return agent.run_strategy_and_get_dataframe(
        buy_condition=buy_condition,
        sell_condition=sell_condition,
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        timeframe=timeframe,
        api_url=api_url,
    )


if __name__ == "__main__":
    agent = StrategyCreatorAgent()
    df = agent.run_strategy_and_get_dataframe(
        buy_condition="EMA(10) > EMA(30)",
        sell_condition="EMA(10) < EMA(30)",
        symbol="BTC",
        from_date="2024-01-01",
        to_date="2025-01-01",
        timeframe="1d",
        api_url="https://97ne32z9yn5u.share.zrok.io",
    )
    df.to_csv("data/btc_data.csv", index=False)