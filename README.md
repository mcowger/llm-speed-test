# LLM Speed Bench

`llm-speed-bench` is a command-line interface (CLI) tool for benchmarking the performance of Large Language Model (LLM) providers that offer an OpenAI-compatible API.

It is designed to provide detailed, actionable data on the output speed and latency characteristics of different models and providers. It measures key performance indicators from the moment a request is sent until the final token of the response is received, with a focus on streaming APIs.

## Features

*   **OpenAI-Compatible:** Works with any API that adheres to the OpenAI specification for streaming chat completions.
*   **Streaming First:** Benchmarks performance by leveraging the provider's streaming API to get detailed timing data.
*   **Detailed Performance Metrics:** Collects and calculates a comprehensive set of metrics, including token counts, time to first token, inter-token latency, and overall throughput.
*   **Flexible Configuration:** Manage inputs via both command-line arguments and environment variables.
*   **Multiple Output Formats:** Presents results in a clean, human-readable format, with an option for machine-readable JSON.

## Installation

There are multiple ways to use `llm-speed-bench`:

### NPX (Recommended)

The easiest way to run the tool without a permanent installation is to use `npx`. This ensures you are always using the latest version.

```bash
npx llm-speed-bench [options]
```

### Global Installation

If you prefer to have the command available globally, you can install it via npm:

```bash
npm install -g llm-speed-bench
```

Once installed, you can run the tool from any directory:

```bash
llm-speed-bench [options]
```

### Local Installation (for Development)

If you want to contribute to the project or modify the code, you can install it locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/llm-speed-bench.git
    cd llm-speed-bench
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Build the project:**
    ```bash
    npm run build
    ```
    This will compile the TypeScript code into JavaScript and place the executable in the `dist/` directory. You can then run the tool directly: `./dist/index.js`.

## Usage

You can run the tool using the compiled executable located at `dist/index.js`. Configuration can be provided through command-line arguments or environment variables.

### Configuration

The tool requires four pieces of information to run:

| Parameter | CLI Argument | Environment Variable | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| API Base URL | `--api-base-url <url>` | `LLM_API_BASE_URL` | Yes | The base URL for the OpenAI-compatible API. |
| API Key | `--api-key <key>` | `LLM_API_KEY` | Yes | The authentication key for the API. |
| Model Name | `--model <name>` | `LLM_MODEL_NAME` | Yes | The specific model to be benchmarked (e.g., `gpt-4o`). |
| Prompt | `--prompt <text>` | `LLM_PROMPT` | Yes | The input text to send to the model. |

### Examples

#### Using Command-Line Arguments

```bash
./dist/index.js \
  --api-base-url "https://api.openai.com/v1" \
  --api-key "sk-..." \
  --model "gpt-4o" \
  --prompt "Tell me a short story about a robot who discovers music."
```

#### Using Environment Variables

You can create a `.env` file in the project root or export the variables in your shell:

**`.env` file:**
```
LLM_API_BASE_URL="https://api.openai.com/v1"
LLM_API_KEY="sk-..."
LLM_MODEL_NAME="gpt-4o"
LLM_PROMPT="Tell me a short story about a robot who discovers music."
```

Then, run the tool:
```bash
./dist/index.js
```

#### Getting JSON Output

To get the results in a machine-readable JSON format, use the `--json` flag:

```bash
./dist/index.js --json > results.json
```

## Output Format

### Standard Output

The default output is a human-readable summary:

```
LLM Benchmark Results
=======================

Configuration
-----------------------
Provider API Base:   https://api.groq.com/openai
Model:               llama3-70b-8192

Metrics
-----------------------
Time to First Token:   152 ms
Total Wall Clock Time: 2,130 ms
Overall Output Rate:   234.7 tokens/sec

Token Counts
-----------------------
Prompt Tokens:         35 (estimated)
Output Tokens:         450

Inter-Token Latency (ms)
-----------------------
Min:                 2 ms
Mean:                4.1 ms
Median:              4 ms
Max:                 15 ms
p90:                 6 ms
p95:                 8 ms
p99:                 12 ms
```

### JSON Output (`--json`)

The JSON output includes all the calculated metrics and configuration details.

```json
{
  "configuration": {
    "apiBaseUrl": "https://api.groq.com/openai",
    "model": "llama3-70b-8192"
  },
  "metrics": {
    "timeToFirstTokenMs": 152,
    "totalWallClockTimeMs": 2130,
    "overallOutputRateTps": 234.7
  },
  "tokenCounts": {
    "promptTokens": 35,
    "outputTokens": 450
  },
  "interTokenLatencyMs": {
    "min": 2,
    "mean": 4.1,
    "median": 4,
    "max": 15,
    "p90": 6,
    "p95": 8,
    "p99": 12
  }
}
```

## Development

### Running with ts-node

To run the tool in development mode without building, you can use `ts-node`:

```bash
npx ts-node src/index.ts --api-base-url ...
```

### Local Installation and Testing

To test the CLI locally as if it were globally installed, you can use `npm link`. This is the best way to test the final command-line experience before publishing.

1.  **Build the project:**
    Make sure your latest changes are compiled.
    ```bash
    npm run build
    ```

2.  **Link the package:**
    This creates a global symbolic link to your local project.
    ```bash
    npm link
    ```

3.  **Run the command globally:**
    You can now run the command from any directory.
    ```bash
    llm-speed-bench --api-base-url "..." --api-key "..."
    ```

4.  **Rebuild after changes:**
    Whenever you change the source code, just re-run the build command. The symbolic link will ensure your global command always uses the latest compiled code.
    ```bash
    npm run build
    ```

5.  **Unlink the package:**
    When you're done with local testing, you can remove the global link.
    ```bash
    npm unlink llm-speed-bench
    ```

```