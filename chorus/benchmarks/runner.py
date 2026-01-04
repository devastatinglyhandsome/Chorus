# Benchmark runner for evaluating chorus performance

import asyncio
import json
import logging
from typing import List, Optional
from dataclasses import dataclass, asdict

from chorus.core.chorus import VotingChorus, ChorusConfig, ChorusResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    name: str
    prompts: List[str]
    expected_answers: List[str] = None


@dataclass
class BenchmarkMetrics:
    prompt: str
    final_answer: str
    total_latency_ms: float
    parallel_latency_ms: float
    num_models_responded: int
    individual_latencies: List[float]
    cost_estimate_usd: float


class BenchmarkRunner:
    def __init__(self, chorus: VotingChorus):
        self.chorus = chorus
        self.results: List[BenchmarkMetrics] = []

    async def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        num_samples: Optional[int] = None,
    ) -> List[BenchmarkMetrics]:
        prompts = dataset.prompts[:num_samples] if num_samples else dataset.prompts

        logger.info(f"Running benchmark on {len(prompts)} prompts")

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")

            try:
                result = await self.chorus.generate(prompt)
                metrics = self._compute_metrics(prompt, result)
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"Failed on prompt {i}: {e}")
                continue

        return self.results

    def _compute_metrics(self, prompt: str, result: ChorusResult) -> BenchmarkMetrics:
        individual_latencies = [
            r.generation_time_ms for r in result.individual_responses
        ]

        total_tokens = sum(r.tokens_generated for r in result.individual_responses)
        cost = (total_tokens / 1000) * 0.0002

        return BenchmarkMetrics(
            prompt=prompt,
            final_answer=result.final_answer,
            total_latency_ms=result.total_latency_ms,
            parallel_latency_ms=result.parallel_latency_ms,
            num_models_responded=len(result.individual_responses),
            individual_latencies=individual_latencies,
            cost_estimate_usd=cost,
        )

    def save_results(self, filepath: str):
        results_dict = [asdict(r) for r in self.results]

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Saved {len(self.results)} results to {filepath}")

    def print_summary(self):
        if not self.results:
            logger.info("No results to summarize")
            return

        avg_latency = sum(r.total_latency_ms for r in self.results) / len(self.results)
        avg_parallel = sum(r.parallel_latency_ms for r in self.results) / len(self.results)
        total_cost = sum(r.cost_estimate_usd for r in self.results)

        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"Total prompts: {len(self.results)}")
        print(f"Avg total latency: {avg_latency:.1f}ms")
        print(f"Avg parallel latency: {avg_parallel:.1f}ms")
        print(f"Total estimated cost: ${total_cost:.4f}")
        print(f"Avg cost per query: ${total_cost/len(self.results):.6f}")
        print("="*50 + "\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run chorus benchmarks")
    parser.add_argument("--models", nargs="+", required=True, help="Chorus models")
    parser.add_argument("--judge", required=True, help="Judge model")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", default="results.json", help="Output file")
    parser.add_argument("--num-samples", type=int, help="Number of samples to run")

    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset_data = json.load(f)

    dataset = BenchmarkDataset(
        name=dataset_data["name"],
        prompts=dataset_data["prompts"],
        expected_answers=dataset_data.get("expected_answers", []),
    )

    config = ChorusConfig(
        chorus_models=args.models,
        judge_model=args.judge,
        num_gpus=args.num_gpus,
    )

    chorus = VotingChorus(config)
    await chorus.initialize()

    runner = BenchmarkRunner(chorus)
    await runner.run_benchmark(dataset, num_samples=args.num_samples)

    runner.save_results(args.output)
    runner.print_summary()

    await chorus.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

