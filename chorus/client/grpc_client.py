# Async gRPC client for communicating with model servers

import asyncio
import grpc
import logging
from typing import Optional, List
from dataclasses import dataclass

from chorus.server.protos import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    logprobs: List[float]
    tokens_generated: int
    generation_time_ms: float
    model_name: str


class AsyncLLMClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[inference_pb2_grpc.LLMInferenceStub] = None

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
        self.stub = inference_pb2_grpc.LLMInferenceStub(self.channel)

        max_retries = 10
        for _ in range(max_retries):
            if await self.health_check():
                logger.info(f"Connected to model server at {self.host}:{self.port}")
                return
            await asyncio.sleep(1)

        raise RuntimeError(f"Failed to connect to server at {self.host}:{self.port}")

    async def health_check(self) -> bool:
        if self.stub is None:
            return False

        try:
            request = inference_pb2.HealthCheckRequest(service="LLMInference")
            response = await self.stub.HealthCheck(request, timeout=5.0)
            return response.status == inference_pb2.HealthCheckResponse.SERVING
        except grpc.RpcError as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        return_logprobs: bool = True,
    ) -> GenerationResult:
        if self.stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        request = inference_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_logprobs=return_logprobs,
        )

        try:
            response = await self.stub.Generate(request, timeout=self.timeout)

            return GenerationResult(
                text=response.text,
                logprobs=list(response.logprobs),
                tokens_generated=response.tokens_generated,
                generation_time_ms=response.generation_time_ms,
                model_name=response.model_name,
            )
        except grpc.RpcError as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def close(self):
        if self.channel:
            await self.channel.close()

