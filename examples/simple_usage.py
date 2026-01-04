# Simple usage example for Chorus

import asyncio
import logging
from chorus.core.chorus import VotingChorus, ChorusConfig

logging.basicConfig(level=logging.INFO)


async def main():
    config = ChorusConfig(
        chorus_models=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        judge_model="Qwen/Qwen2.5-1.5B-Instruct",
        num_gpus=1,
    )

    chorus = VotingChorus(config)
    await chorus.initialize()

    prompt = "Explain quantum entanglement in simple terms."
    result = await chorus.generate(prompt)

    print(f"Final Answer: {result.final_answer}")
    print(f"Latency: {result.total_latency_ms:.0f}ms")
    print(f"Judge Reasoning: {result.judge_reasoning}")

    await chorus.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
