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

interface TokenCounts {
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
  totalOutputTokens: number;
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
}

interface ViewpointMetrics {
  outputTokens: number;
  timeToFirstTokenMs: number;
  timeToSecondTokenMs: number;
  outputRateTpsFromSecondToken: number;
  outputRateTpsFromRequestStart: number;
  interTokenLatencyMs: InterTokenLatency;
}

interface Metrics {
  totalWallClockTimeMs: number;
  timeToSecondTokenMs: number;
  views: {
    reasoningInclusive: ViewpointMetrics;
    reasoningOnly: ViewpointMetrics;
    completionOnly: ViewpointMetrics;
  };
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
  let T_second_token: number | null = null;

  const tokenTimestamps = {
    all: [] as number[],
    completion: [] as number[],
    reasoning: [] as number[],
  };

  let totalOutputTokens = 0;
  let completionTokens = 0;
  let reasoningTokens = 0;

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

      let chunkProducedTokens = false;
      let chunkProducedCompletionTokens = 0;
      let chunkProducedReasoningTokens = 0;
      for (const line of lines) {
        const data = line.substring(5).trim();
        if (data === '[DONE]') {
          break;
        }
        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta;
          if (!delta) {
            continue;
          }
          const completionCount = countTokenLikeEntries(delta.content);
          const reasoningCount = countTokenLikeEntries(delta.reasoning);
          const reasoningContentCount = countTokenLikeEntries((delta as Record<string, unknown>).reasoning_content);
          const totalReasoning = reasoningCount + reasoningContentCount;
          if (completionCount > 0 || totalReasoning > 0) {
            chunkProducedTokens = true;
            chunkProducedCompletionTokens += completionCount;
            chunkProducedReasoningTokens += totalReasoning;
          }
        } catch (e) {
          // Ignore parsing errors for non-JSON lines
        }
      }

      if (chunkProducedTokens) {
        const chunkTime = performance.now();

        const producedTokens = chunkProducedCompletionTokens + chunkProducedReasoningTokens;
        totalOutputTokens += producedTokens;
        completionTokens += chunkProducedCompletionTokens;
        reasoningTokens += chunkProducedReasoningTokens;

        const registerTimestamp = (container: number[], count: number) => {
          for (let i = 0; i < count; i++) {
            container.push(chunkTime);
          }
        };

        if (chunkProducedCompletionTokens > 0) {
          registerTimestamp(tokenTimestamps.completion, chunkProducedCompletionTokens);
          registerTimestamp(tokenTimestamps.all, chunkProducedCompletionTokens);
        }

        if (chunkProducedReasoningTokens > 0) {
          registerTimestamp(tokenTimestamps.reasoning, chunkProducedReasoningTokens);
          registerTimestamp(tokenTimestamps.all, chunkProducedReasoningTokens);
        }

        if (tokenTimestamps.all.length >= 2 && T_second_token === null) {
          T_second_token = tokenTimestamps.all[1]!;
        }

        const totalTokensLabel = totalOutputTokens === 1 ? 'token' : 'tokens';
        const completionLabel = completionTokens === 1 ? 'token' : 'tokens';
        const reasoningLabel = reasoningTokens === 1 ? 'token' : 'tokens';

        let message = `Receiving Tokens... (total: ${totalOutputTokens} ${totalTokensLabel}`;
        message += ` | completion: ${completionTokens} ${completionLabel}`;
        message += ` | reasoning: ${reasoningTokens} ${reasoningLabel}`;

        if (T_second_token && totalOutputTokens > 1) {
          const elapsedS = (chunkTime - T_second_token) / 1000;
          const tokensSinceSecond = totalOutputTokens - 1;
          const tokensPerSecond = elapsedS > 0 ? tokensSinceSecond / elapsedS : 0;
          message += ` @ ${tokensPerSecond.toFixed(2)} tokens/s`;
        }
        message += ')';
        progress.update(message);
      }
    }

    const T_end = performance.now();
    progress.stop('Finished receiving tokens.');
    const totalWallClockTimeMs = T_end - T_start;

    // Estimate prompt tokens using tiktoken
    const encoding = get_encoding('cl100k_base');
    const promptTokens = encoding.encode(prompt).length;
    encoding.free();

    const inclusiveMetrics = computeViewpointMetrics(
      tokenTimestamps.all,
      totalOutputTokens,
      totalWallClockTimeMs,
      T_start,
      T_end,
    );

    const reasoningOnlyMetrics = computeViewpointMetrics(
      tokenTimestamps.reasoning,
      reasoningTokens,
      totalWallClockTimeMs,
      T_start,
      T_end,
    );

    const completionOnlyMetrics = computeViewpointMetrics(
      tokenTimestamps.completion,
      completionTokens,
      totalWallClockTimeMs,
      T_start,
      T_end,
    );

    const results: BenchmarkResults = {
      configuration: {
        apiBaseUrl,
        model,
      },
      metrics: {
        totalWallClockTimeMs: parseFloat(totalWallClockTimeMs.toFixed(2)),
        timeToSecondTokenMs: parseFloat((T_second_token ? T_second_token - T_start : 0).toFixed(2)),
        views: {
          reasoningInclusive: inclusiveMetrics,
          reasoningOnly: reasoningOnlyMetrics,
          completionOnly: completionOnlyMetrics,
        },
      },
      tokenCounts: {
        promptTokens,
        completionTokens,
        reasoningTokens,
        totalOutputTokens,
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
      metrics: {
        ...results.metrics,
        views: {
          reasoningInclusive: {
            ...results.metrics.views.reasoningInclusive,
            interTokenLatencyMs: {
              ...results.metrics.views.reasoningInclusive.interTokenLatencyMs,
              latencies: undefined,
            },
          },
          reasoningOnly: {
            ...results.metrics.views.reasoningOnly,
            interTokenLatencyMs: {
              ...results.metrics.views.reasoningOnly.interTokenLatencyMs,
              latencies: undefined,
            },
          },
          completionOnly: {
            ...results.metrics.views.completionOnly,
            interTokenLatencyMs: {
              ...results.metrics.views.completionOnly.interTokenLatencyMs,
              latencies: undefined,
            },
          },
        },
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
  console.log(`Total Wall Clock Time: ${results.metrics.totalWallClockTimeMs} ms (${ms(results.metrics.totalWallClockTimeMs)})`);
  console.log(`Time to Second Token (all tokens): ${results.metrics.timeToSecondTokenMs} ms (${ms(results.metrics.timeToSecondTokenMs)})`);
  console.log();

  console.log('Token Counts');
  console.log('-----------------------');
  console.log(`Prompt Tokens:         ${results.tokenCounts.promptTokens} (estimated)`);
  console.log(`Reasoning Tokens:      ${results.tokenCounts.reasoningTokens}`);
  console.log(`Completion Tokens:     ${results.tokenCounts.completionTokens}`);
  console.log(`Total Output Tokens:   ${results.tokenCounts.totalOutputTokens}`);
  console.log();

  const { reasoningInclusive, reasoningOnly, completionOnly } = results.metrics.views;

  console.log('Reasoning + Completion View');
  console.log('-----------------------');
  console.log(`Time to Second Token:             ${reasoningInclusive.timeToSecondTokenMs} ms (${ms(reasoningInclusive.timeToSecondTokenMs)})`);
  console.log(`Time to Second Reasoning Token:   ${reasoningOnly.timeToSecondTokenMs} ms (${ms(reasoningOnly.timeToSecondTokenMs)})`);
  console.log(`Time to Second Completion Token:  ${completionOnly.timeToSecondTokenMs} ms (${ms(completionOnly.timeToSecondTokenMs)})`);
  console.log(`Output Rate (since 2nd reasoning token): ${reasoningOnly.outputRateTpsFromSecondToken} tokens/sec`);
  console.log(`Output Rate (since 2nd completion token): ${completionOnly.outputRateTpsFromSecondToken} tokens/sec`);
  console.log(`Output Rate (wall clock):              ${reasoningInclusive.outputRateTpsFromRequestStart} tokens/sec`);
  console.log();

  console.log('Inter-Token Latency (ms)');
  console.log('-----------------------');
  console.log(`Min:                 ${reasoningInclusive.interTokenLatencyMs.min} ms`);
  console.log(`Mean:                ${reasoningInclusive.interTokenLatencyMs.mean} ms`);
  console.log(`Median:              ${reasoningInclusive.interTokenLatencyMs.median} ms`);
  console.log(`Max:                 ${reasoningInclusive.interTokenLatencyMs.max} ms`);
  console.log(`p90:                 ${reasoningInclusive.interTokenLatencyMs.p90} ms`);
  console.log(`p95:                 ${reasoningInclusive.interTokenLatencyMs.p95} ms`);
  console.log(`p99:                 ${reasoningInclusive.interTokenLatencyMs.p99} ms`);
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

/**
 * Counts token-like entries within a streamed delta field.
 * @param value - The value to examine for token content.
 * @returns The number of token-like entries discovered.
 */
function countTokenLikeEntries(value: unknown): number {
  if (value === null || value === undefined) {
    return 0;
  }

  if (typeof value === 'string') {
    return value.trim() ? 1 : 0;
  }

  if (Array.isArray(value)) {
    return value.reduce((total, item) => total + countTokenLikeEntries(item), 0);
  }

  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    if (obj.tokens !== undefined) {
      return countTokenLikeEntries(obj.tokens);
    }
    if (obj.content !== undefined) {
      return countTokenLikeEntries(obj.content);
    }
    if (typeof obj.text === 'string') {
      return obj.text.trim() ? 1 : 0;
    }
    return 0;
  }

  return 0;
}

/**
 * Computes throughput and latency statistics for a given set of token timestamps.
 * @param timestamps - The timestamps for each token in the view.
 * @param tokenCount - The total number of tokens in the view.
 * @param totalWallClockTimeMs - The total wall clock time of the run.
 * @param start - The benchmark start time.
 * @param end - The benchmark end time.
 * @returns Metrics describing the selected viewpoint.
 */
function computeViewpointMetrics(
  timestamps: number[],
  tokenCount: number,
  totalWallClockTimeMs: number,
  start: number,
  end: number,
): ViewpointMetrics {
  const timeToFirstTokenMs = timestamps[0] !== undefined ? timestamps[0] - start : 0;
  const timeToSecondTokenMs = timestamps[1] !== undefined ? timestamps[1] - start : 0;

  const tokensExcludingFirst = Math.max(tokenCount - 1, 0);
  const outputDurationMsFromSecondToken = timestamps[1] !== undefined ? end - timestamps[1]! : 0;
  const outputRateTpsFromSecondToken = outputDurationMsFromSecondToken > 0
    ? (tokensExcludingFirst / outputDurationMsFromSecondToken) * 1000
    : 0;
  const outputRateTpsFromRequestStart = totalWallClockTimeMs > 0
    ? (tokensExcludingFirst / totalWallClockTimeMs) * 1000
    : 0;

  const latencies: number[] = [];
  for (let i = 1; i < timestamps.length; i++) {
    const previous = timestamps[i - 1];
    const current = timestamps[i];
    if (previous !== undefined && current !== undefined) {
      latencies.push(current - previous);
    }
  }
  const latencyStats = calculateLatencyStats(latencies);

  return {
    outputTokens: tokenCount,
    timeToFirstTokenMs: parseFloat(timeToFirstTokenMs.toFixed(2)),
    timeToSecondTokenMs: parseFloat(timeToSecondTokenMs.toFixed(2)),
    outputRateTpsFromSecondToken: parseFloat(outputRateTpsFromSecondToken.toFixed(2)),
    outputRateTpsFromRequestStart: parseFloat(outputRateTpsFromRequestStart.toFixed(2)),
    interTokenLatencyMs: {
      ...latencyStats,
      latencies,
    },
  };
}
