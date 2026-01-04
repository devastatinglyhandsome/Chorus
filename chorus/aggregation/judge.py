# LLM-as-judge aggregation for combining multiple model responses

import logging
from typing import List, Tuple

from chorus.client.grpc_client import AsyncLLMClient, GenerationResult

logger = logging.getLogger(__name__)


class JudgeAggregator:
    def __init__(self, judge_client: AsyncLLMClient):
        self.judge_client = judge_client

    def _create_judge_prompt(self, original_prompt: str, responses: List[GenerationResult]) -> str:
        prompt_parts = [
            "You are an expert judge evaluating multiple AI-generated responses.",
            "Your task: Select the BEST response or synthesize a better answer.",
            "",
            f"Original Question: {original_prompt}",
            "",
            "Candidate Responses:",
        ]

        for i, response in enumerate(responses, 1):
            prompt_parts.append(f"\n=== Response {i} (from {response.model_name}) ===")
            prompt_parts.append(response.text)

        prompt_parts.extend([
            "",
            "Instructions:",
            "1. Evaluate each response for accuracy, completeness, and clarity",
            "2. Check for consistency across responses",
            "3. Select the best response OR synthesize a better one",
            "4. Provide your reasoning",
            "",
            "Format your answer as:",
            "REASONING: <explain your evaluation>",
            "FINAL ANSWER: <the best or synthesized response>",
        ])

        return "\n".join(prompt_parts)

    async def aggregate(self, prompt: str, responses: List[GenerationResult]) -> Tuple[str, str]:
        judge_prompt = self._create_judge_prompt(prompt, responses)

        logger.info("Sending to judge for aggregation")

        judge_result = await self.judge_client.generate(
            prompt=judge_prompt,
            max_tokens=1024,
            temperature=0.3,
            return_logprobs=False,
        )

        judge_text = judge_result.text

        try:
            reasoning_part = judge_text.split("REASONING:")[1].split("FINAL ANSWER:")[0].strip()
            answer_part = judge_text.split("FINAL ANSWER:")[1].strip()
        except IndexError:
            logger.warning("Judge didn't follow expected format, using full response")
            reasoning_part = "Judge response not in expected format"
            answer_part = judge_text

        return answer_part, reasoning_part

