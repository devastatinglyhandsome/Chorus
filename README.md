# Chorus

Aggregating outputs from multiple small models to match or exceed single large model quality with lower latency and cost.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import asyncio
from chorus.core.chorus import VotingChorus, ChorusConfig

async def main():
    config = ChorusConfig(
        chorus_models=[
            "meta-llama/Llama-3.2-3b",
            "Qwen/Qwen2.5-3B",
        ],
        judge_model="meta-llama/Llama-3.2-1b",
        num_gpus=2,
    )
    
    chorus = VotingChorus(config)
    await chorus.initialize()
    
    result = await chorus.generate("Explain quantum entanglement.")
    print(result.final_answer)
    
    await chorus.shutdown()

asyncio.run(main())
```

## Configuration

Copy `.env.example` to `.env` and configure your models and GPU settings.

## Features

- Parallel execution of multiple LLMs
- LLM-as-judge aggregation
- Flexible GPU configuration
- Benchmarking and performance dashboards

## Requirements

- Python >= 3.9
- CUDA-capable GPU(s)
- vLLM compatible models
