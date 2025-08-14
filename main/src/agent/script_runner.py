import subprocess
import tempfile
import os
import pandas as pd
from io import StringIO
from typing import Union, Tuple

def run_sandboxed_python_script(script_code: str, timeout_seconds: int = 30, return_dataframe: bool = True):
    """
    Executes a Python script in a secure Docker container sandbox.
    
    Args:
        script_code: The Python code to execute as a string.
        timeout_seconds: The maximum time (in seconds) to allow the script to run.
        return_dataframe: If True, attempts to parse stdout as CSV and return DataFrame.

    Returns:
        If return_dataframe=True and successful:
            - DataFrame or stdout (Union[pd.DataFrame, str])
            - stderr (str)
            - exit_code (int)
        Otherwise:
            - stdout (str)
            - stderr (str) 
            - exit_code (int)
    """
    # Use a temporary file to store the script code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(script_code)
        temp_script_path = temp_script.name

    try:
        # Construct the Docker command
        # --rm: Remove the container after it exits.
        # -v: Mount the temporary script file into the container.
        # my_python_sandbox: The name of our custom Docker image.
        # python /app/script.py: The command to execute inside the container.
        # We also pass resource limits here for security and stability.
        docker_command = [
            'docker', 'run', '--rm', 
            '--cpus', '1', '--memory', '512m',
            '-v', f"{temp_script_path}:/test_script.py",
            'py_sandbox',
            'python', '/test_script.py'
        ]

        # Execute the command using subprocess.run
        # capture_output=True: Captures stdout and stderr.
        # text=True: Decodes the output as text (strings).
        # timeout: Kills the process if it runs longer than the specified time.
        result = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        # Try to parse as DataFrame if requested and successful
        if return_dataframe and exit_code == 0 and stdout.strip():
            try:
                df = pd.read_csv(StringIO(stdout))
                return df, stderr, exit_code
            except Exception:
                # Fall back to raw stdout if CSV parsing fails
                return stdout, stderr, exit_code
        
        return stdout, stderr, exit_code

    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"Error: Script execution timed out after {timeout_seconds} seconds."
        exit_code = -1
    except FileNotFoundError:
        stdout = ""
        stderr = "Error: Docker command not found. Is Docker installed and in your PATH?"
        exit_code = -2
    except Exception as e:
        stdout = ""
        stderr = f"An unexpected error occurred: {str(e)}"
        exit_code = -3
    finally:
        # Crucial cleanup step: delete the temporary script file
        os.remove(temp_script_path)

    return stdout, stderr, exit_code

if __name__ == "__main__":
    # --- Example 1: Successful script execution with DataFrame output ---
    print("--- Example 1: Successful Execution with DataFrame ---")
    good_script = """
import sys
import requests
import pandas as pd
import numpy as np

def fetch_data():
    url = "https://tender-poets-play.loca.lt/data"
    payload = {"name":"BTC","from_date":"2024-01-01","to_date":"2025-01-01","timeframe":"1d"}
    headers = {"Content-Type":"application/json"}
    try:
        r = requests.post(url,json=payload,headers=headers,timeout=10)
    except requests.RequestException as e:
        print(f"Error: Unable to connect to API - {e}", file=sys.stderr)
        sys.exit(1)
    if r.status_code != 200:
        print(f"Error: API returned status code {r.status_code}", file=sys.stderr)
        sys.exit(1)
    try:
        jd = r.json()
    except ValueError:
        print("Error: Response is not valid JSON", file=sys.stderr)
        sys.exit(1)
    if not jd.get("success"):
        print("Error: API response indicates failure", file=sys.stderr)
        sys.exit(1)
    data = jd.get("data")
    if not data or not isinstance(data,list):
        print("Error: No data found in API response", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(data)
    if df.empty:
        print("Error: DataFrame is empty after loading data", file=sys.stderr)
        sys.exit(1)
    return df

def generate_signals(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["SMA_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["SMA_30"] = df["close"].rolling(window=30, min_periods=1).mean()
    df["buy_sell_signal"] = 0
    diff = df["SMA_10"] - df["SMA_30"]
    prev_diff = diff.shift(1)
    df.loc[(diff > 0) & (prev_diff <= 0), "buy_sell_signal"] = 1
    df.loc[(diff < 0) & (prev_diff >= 0), "buy_sell_signal"] = -1
    df.drop(["SMA_10","SMA_30"], axis=1, inplace=True)
    return df

def main():
    df = fetch_data()
    df = generate_signals(df)
    final_df = pd.DataFrame({
        "security_name": df["symbol"],
        "datetime": df["timestamp"],
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "buy_sell_signal": df["buy_sell_signal"]
    })
    print(final_df.to_csv(index=False))

if __name__=="__main__":
    main()
"""
    result, stderr, exit_code = run_sandboxed_python_script(good_script)
    
    if isinstance(result, pd.DataFrame):
        print("Successfully parsed as DataFrame:")
        print(result)
        print(f"DataFrame shape: {result.shape}")
        print(f"DataFrame columns: {list(result.columns)}")
    else:
        print("Raw output:", result)
    print("STDERR:", stderr)
    print("Exit Code:", exit_code)

    # --- Example 2: Script with an error ---
    print("\n--- Example 2: Script with an Error ---")
    bad_script = """
import pandas as pd
# This will cause a NameError because 'np' is not imported
df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
print(df.to_csv(index=False))
"""
    result, stderr, exit_code = run_sandboxed_python_script(bad_script)
    
    if isinstance(result, pd.DataFrame):
        print("Successfully parsed as DataFrame:")
        print(result)
    else:
        print("Raw output:", result)
    print("STDERR:", stderr)
    print("Exit Code:", exit_code)

    # --- Example 3: Script that exceeds the timeout ---
    print("\n--- Example 3: Script Timeout ---")
    infinite_loop_script = """
import time
print("Starting infinite loop...")
while True:
    time.sleep(1)
"""
    result, stderr, exit_code = run_sandboxed_python_script(infinite_loop_script, timeout_seconds=20)
    
    if isinstance(result, pd.DataFrame):
        print("Successfully parsed as DataFrame:")
        print(result)
    else:
        print("Raw output:", result)
    print("STDERR:", stderr)
    print("Exit Code:", exit_code)

    # --- Example 4: Force raw output (no DataFrame parsing) ---
    print("\n--- Example 4: Raw Output (No DataFrame Parsing) ---")
    raw_script = """
print("Hello, World!")
print("This is raw text output")
"""
    result, stderr, exit_code = run_sandboxed_python_script(raw_script, return_dataframe=False)
    
    if isinstance(result, pd.DataFrame):
        print("Successfully parsed as DataFrame:")
        print(result)
    else:
        print("Raw output:", result)
    print("STDERR:", stderr)
    print("Exit Code:", exit_code)