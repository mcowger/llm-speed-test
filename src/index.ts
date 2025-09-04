#!/usr/bin/env node
process.env.DOTENV_CONFIG_SILENT = 'true';

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import dotenv from 'dotenv';
import { get_encoding } from 'tiktoken';
import ms from 'ms';

// Load environment variables from .env file
dotenv.config({ quiet: true });

// --- Type Definitions ---

interface Config {
  apiBaseUrl: string;
  apiKey: string;
  model: string;
  prompt: string;
  json: boolean;
  effort: 'minimal' | 'low' | 'medium' | 'high' | undefined;
}

interface Metrics {
  timeToFirstTokenMs: number;
  totalWallClockTimeMs: number;
  outputRateTpsFromFirstToken: number;
  outputRateTpsFromRequestStart: number;
}

interface TokenCounts {
  promptTokens: number;
  outputTokens: number;
}

interface LatencyStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  p90: number;
  p95: number;
  p99: number;
}

interface InterTokenLatency extends LatencyStats {
  latencies: number[];
}

interface BenchmarkResults {
  configuration: {
    apiBaseUrl: string;
    model: string;
  };
  metrics: Metrics;
  tokenCounts: TokenCounts;
  interTokenLatencyMs: InterTokenLatency;
}

// --- Utility Functions ---

/**
 * Calculates percentile values from a sorted array of numbers.
 * @param arr - A sorted array of numbers.
 * @param p - The percentile to calculate (0-100).
 * @returns The value at the given percentile.
 */
const percentile = (arr: number[], p: number): number => {
  if (arr.length === 0) return 0;
  const k = (p / 100) * (arr.length - 1);
  const f = Math.floor(k);
  const c = Math.ceil(k);
  if (f === c) {
    return arr[f]!;
  }
  const fVal = arr[f]!;
  const cVal = arr[c]!;
  return fVal * (c - k) + cVal * (k - f);
};

/**
 * Calculates statistical metrics for a list of latencies.
 * @param latencies - An array of numbers.
 * @returns An object with min, max, mean, median, and percentile stats.
 */
const calculateLatencyStats = (latencies: number[]): LatencyStats => {
  if (latencies.length === 0) {
    return { min: 0, max: 0, mean: 0, median: 0, p90: 0, p95: 0, p99: 0 };
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const mean = sum / sorted.length;
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  if (min === undefined || max === undefined) {
    // This path should be unreachable due to the length check above,
    // but it satisfies the strict type checker.
    return { min: 0, max: 0, mean: 0, median: 0, p90: 0, p95: 0, p99: 0 };
  }

  const median = percentile(sorted, 50);
  const p90 = percentile(sorted, 90);
  const p95 = percentile(sorted, 95);
  const p99 = percentile(sorted, 99);

  return {
    min: parseFloat(min.toFixed(2)),
    max: parseFloat(max.toFixed(2)),
    mean: parseFloat(mean.toFixed(2)),
    median: parseFloat(median.toFixed(2)),
    p90: parseFloat(p90.toFixed(2)),
    p95: parseFloat(p95.toFixed(2)),
    p99: parseFloat(p99.toFixed(2)),
  };
};

// --- Main Application Logic ---

class ProgressIndicator {
  private stream: NodeJS.WriteStream = process.stderr;
  private timer: NodeJS.Timeout | undefined;
  private spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  private frame = 0;
  private message = '';

  constructor(private isJson: boolean) {}

  private render() {
    if (this.isJson) return;
    this.stream.clearLine(0);
    this.stream.cursorTo(0);
    const spinnerFrame = this.spinner[this.frame];
    this.stream.write(`${spinnerFrame} ${this.message}`);
    this.frame = (this.frame + 1) % this.spinner.length;
  }

  start(message: string) {
    if (this.isJson) return;
    this.message = message;
    this.timer = setInterval(() => this.render(), 80);
  }

  update(message: string) {
    if (this.isJson) return;
    this.message = message;
  }

  stop(finalMessage: string) {
    if (this.isJson) return;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    this.stream.clearLine(0);
    this.stream.cursorTo(0);
    this.stream.write(`✅ ${finalMessage}\n`);
  }

  clear() {
    if (this.isJson) return;
    this.stream.clearLine(0);
    this.stream.cursorTo(0);
  }
}

/**
 * Parses command-line arguments and environment variables.
 * @returns The configuration object.
 */
async function getConfig(): Promise<Config> {
  const argv = await yargs(hideBin(process.argv))
    .option('api-base-url', {
      alias: 'u',
      type: 'string',
      description: 'The base URL for the OpenAI-compatible API endpoint.',
      default: process.env.LLM_API_URL,
    })
    .option('api-key', {
      alias: 'k',
      type: 'string',
      description: 'The authentication key for the API.',
      default: process.env.LLM_API_KEY,
    })
    .option('model', {
      alias: 'm',
      type: 'string',
      description: 'The specific model to be benchmarked.',
      default: process.env.LLM_MODEL_NAME,
    })
    .option('prompt', {
      alias: 'p',
      type: 'string',
      description: 'The input text to send to the model.',
      default: process.env.LLM_PROMPT,
    })
    .option('json', {
      type: 'boolean',
      description: 'Output the results in JSON format.',
      default: false,
    })
    .option('effort', {
      alias: 'e',
      type: 'string',
      description: 'Reasoning effort level.',
      choices: ['minimal', 'low', 'medium', 'high'] as const,
    })
    .help()
    .alias('help', 'h')
    .parse();

  if (!argv.apiBaseUrl || !argv.apiKey || !argv.model || !argv.prompt) {
    console.error('Error: Missing required arguments.');
    console.error('Please provide api-base-url, api-key, model, and prompt via CLI arguments or environment variables.');
    // Throwing an error is cleaner and more explicit for type checkers
    throw new Error('Missing required arguments.');
  }

  const config: Config = {
    apiBaseUrl: argv.apiBaseUrl as string,
    apiKey: argv.apiKey as string,
    model: argv.model as string,
    prompt: argv.prompt as string,
    json: argv.json ?? false,
    effort: argv.effort,
  };
  return config;
}

/**
 * Runs the benchmark against the specified LLM provider.
 * @param config - The configuration object.
 */
async function runBenchmark(config: Config): Promise<void> {
  const { apiBaseUrl, apiKey, model, prompt, json, effort } = config;
  const progress = new ProgressIndicator(json);

  progress.start('Connecting...');
  const T_start = performance.now();
  let T_first_token: number | null = null;

  const interTokenLatencies: number[] = [];
  let outputTokens = 0;
  let lastTokenChunkTime: number | null = null;

  try {
    const url = new URL(apiBaseUrl);
    if (!url.pathname.endsWith('/')) {
      url.pathname += '/';
    }

    let finalUrl;
    if (url.pathname.endsWith('/v1/')) {
        finalUrl = new URL('chat/completions', url);
    } else {
        finalUrl = new URL('v1/chat/completions', url);
    }

    const body: {
      model: string;
      messages: { role: string; content: string }[];
      stream: boolean;
      reasoning?: {
        effort: 'minimal' | 'low' | 'medium' | 'high';
      };
    } = {
      model,
      messages: [{ role: 'user', content: prompt }],
      stream: true,
    };

    if (effort) {
      body.reasoning = { effort };
    }

    progress.update('Sending Request...');
    const response = await fetch(finalUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok || !response.body) {
      const errorBody = await response.text();
      throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorBody}`);
    }

    progress.update('Waiting for First Token...');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter((line) => line.trim().startsWith('data:'));

      let chunkContainsToken = false;
      for (const line of lines) {
        const data = line.substring(5).trim();
        if (data === '[DONE]') {
          break;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.choices[0]?.delta?.content) {
            chunkContainsToken = true;
            outputTokens++;
          }
        } catch (e) {
          // Ignore parsing errors for non-JSON lines
        }
      }

      if (chunkContainsToken) {
        const chunkTime = performance.now();

        if (T_first_token === null) {
          T_first_token = chunkTime;
        }

        if (lastTokenChunkTime !== null) {
          interTokenLatencies.push(chunkTime - lastTokenChunkTime);
        }
        lastTokenChunkTime = chunkTime;

        if (T_first_token) {
            const elapsedS = (performance.now() - T_first_token) / 1000;
            const tokensPerSecond = elapsedS > 0 ? outputTokens / elapsedS : 0;
            progress.update(`Receiving Tokens... (${outputTokens} tokens received @ ${tokensPerSecond.toFixed(2)} tokens/s)`);
        }
      }
    }

    const T_end = performance.now();
    progress.stop('Finished receiving tokens.');
    const totalWallClockTimeMs = T_end - T_start;
    const timeToFirstTokenMs = T_first_token ? T_first_token - T_start : 0;

    // Estimate prompt tokens using tiktoken
    const encoding = get_encoding('cl100k_base');
    const promptTokens = encoding.encode(prompt).length;
    encoding.free();

    const outputDurationMs = totalWallClockTimeMs - timeToFirstTokenMs;
    const outputRateTpsFromFirstToken = outputDurationMs > 0 ? (outputTokens / outputDurationMs) * 1000 : 0;
    const outputRateTpsFromRequestStart = totalWallClockTimeMs > 0 ? (outputTokens / totalWallClockTimeMs) * 1000 : 0;

    const latencyStats = calculateLatencyStats(interTokenLatencies);

    const results: BenchmarkResults = {
      configuration: {
        apiBaseUrl,
        model,
      },
      metrics: {
        timeToFirstTokenMs: parseFloat(timeToFirstTokenMs.toFixed(2)),
        totalWallClockTimeMs: parseFloat(totalWallClockTimeMs.toFixed(2)),
        outputRateTpsFromFirstToken: parseFloat(outputRateTpsFromFirstToken.toFixed(2)),
        outputRateTpsFromRequestStart: parseFloat(outputRateTpsFromRequestStart.toFixed(2)),
      },
      tokenCounts: {
        promptTokens,
        outputTokens,
      },
      interTokenLatencyMs: {
        ...latencyStats,
        latencies: interTokenLatencies,
      },
    };

    printResults(results, config.json);

  } catch (error) {
    console.error('An error occurred during the benchmark:', error);
    process.exit(1);
  }
}


/**
 * Prints the benchmark results to the console.
 * @param results - The benchmark results object.
 * @param asJson - Whether to print in JSON format.
 */
function printResults(results: BenchmarkResults, asJson: boolean): void {
  if (asJson) {
    // Create a new object for JSON output that doesn't include the full latencies array
    const jsonOutput = {
      ...results,
      interTokenLatencyMs: {
        ...results.interTokenLatencyMs,
        latencies: undefined, // Remove the detailed list for the final JSON output
      },
    };
    // delete jsonOutput.interTokenLatencyMs.latencies;
    console.log(JSON.stringify(jsonOutput, null, 2));
    return;
  }

  console.log('LLM Benchmark Results');
  console.log('=======================');
  console.log();

  console.log('Configuration');
  console.log('-----------------------');
  console.log(`Provider API Base:   ${results.configuration.apiBaseUrl}`);
  console.log(`Model:               ${results.configuration.model}`);
  console.log();

  console.log('Metrics');
  console.log('-----------------------');
  console.log(`Time to First Token:   ${results.metrics.timeToFirstTokenMs} ms (${ms(results.metrics.timeToFirstTokenMs)})`);
  console.log(`Total Wall Clock Time: ${results.metrics.totalWallClockTimeMs} ms (${ms(results.metrics.totalWallClockTimeMs)})`);
  console.log(`Output Rate (since 1st token): ${results.metrics.outputRateTpsFromFirstToken} tokens/sec`);
  console.log(`Output Rate (wall clock):      ${results.metrics.outputRateTpsFromRequestStart} tokens/sec`);
  console.log();

  console.log('Token Counts');
  console.log('-----------------------');
  console.log(`Prompt Tokens:         ${results.tokenCounts.promptTokens} (estimated)`);
  console.log(`Output Tokens:         ${results.tokenCounts.outputTokens}`);
  console.log();

  console.log('Inter-Token Latency (ms)');
  console.log('-----------------------');
  console.log(`Min:                 ${results.interTokenLatencyMs.min} ms`);
  console.log(`Mean:                ${results.interTokenLatencyMs.mean} ms`);
  console.log(`Median:              ${results.interTokenLatencyMs.median} ms`);
  console.log(`Max:                 ${results.interTokenLatencyMs.max} ms`);
  console.log(`p90:                 ${results.interTokenLatencyMs.p90} ms`);
  console.log(`p95:                 ${results.interTokenLatencyMs.p95} ms`);
  console.log(`p99:                 ${results.interTokenLatencyMs.p99} ms`);
}

// --- Entry Point ---

(async () => {
  try {
    const config = await getConfig();
    await runBenchmark(config);
  } catch (error) {
    // Catch errors from getConfig and exit gracefully
    // The error message is already printed inside getConfig
    process.exit(1);
  }
})();
