# Main chorus orchestrator for parallel execution and aggregation

import warnings
warnings.filterwarnings('ignore', message=".*MessageFactory.*GetPrototype.*")

import asyncio
import logging
import multiprocessing as mp
import time
from typing import List, Optional
from dataclasses import dataclass

from chorus.core.gpu_manager import GPUManager, ModelAllocation
from chorus.client.grpc_client import AsyncLLMClient, GenerationResult
from chorus.aggregation.judge import JudgeAggregator
from chorus.server.grpc_server import serve

logger = logging.getLogger(__name__)


@dataclass
class ChorusConfig:
    chorus_models: List[str]
    judge_model: str
    num_gpus: int
    on_insufficient_memory: str = "quantize"
    grpc_port_base: int = 50051
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class ChorusResult:
    final_answer: str
    individual_responses: List[GenerationResult]
    judge_reasoning: Optional[str]
    total_latency_ms: float
    parallel_latency_ms: float


class VotingChorus:
    def __init__(self, config: ChorusConfig):
        self.config = config
        self.gpu_manager: Optional[GPUManager] = None
        self.allocations: List[ModelAllocation] = []
        self.chorus_clients: List[AsyncLLMClient] = []
        self.judge_client: Optional[AsyncLLMClient] = None
        self.judge_aggregator: Optional[JudgeAggregator] = None
        self._server_processes: List[mp.Process] = []

    async def initialize(self):
        logger.info("Initializing VotingChorus...")

        self.gpu_manager = GPUManager(
            num_gpus=self.config.num_gpus,
            on_insufficient_memory=self.config.on_insufficient_memory
        )
        self.gpu_manager.detect_gpus()

        self.allocations = self.gpu_manager.plan_allocation(
            chorus_models=self.config.chorus_models,
            judge_model=self.config.judge_model,
        )

        logger.info(f"Planned allocation for {len(self.allocations)} models")

        await self._start_servers()
        await self._create_clients()

        self.judge_aggregator = JudgeAggregator(self.judge_client)

        logger.info("VotingChorus initialized successfully")

    async def _start_servers(self):
        import socket
        import sys
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        processes = []

        for i, allocation in enumerate(self.allocations):
            port = self.config.grpc_port_base + i

            process = mp.Process(
                target=serve,
                args=(allocation, port),
                daemon=False,
            )
            process.start()
            processes.append(process)

            logger.info(
                f"Started server for {allocation.model_name} on port {port} "
                f"(GPU {allocation.gpu_id}, PID: {process.pid})"
            )

        logger.info("Waiting for servers to initialize (this may take 3-5 minutes for model loading on first run)...")
        
        def check_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                return result == 0
            except:
                return False

        max_wait = 300
        check_interval = 5
        for attempt in range(max_wait // check_interval):
            await asyncio.sleep(check_interval)
            
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count < len(processes):
                dead = [p for p in processes if not p.is_alive()]
                for p in dead:
                    logger.error(f"Server process {p.pid} died with exit code {p.exitcode}")
                    logger.error(f"Process args: {p.args}")
                raise RuntimeError(f"{len(dead)} server process(es) died during startup. Check logs above for errors.")
            
            ports_ready = sum(1 for i in range(len(self.allocations)) 
                             if check_port(self.config.grpc_port_base + i))
            
            elapsed = (attempt + 1) * check_interval
            if attempt % 6 == 0 or ports_ready > 0:
                logger.info(f"Waiting... {ports_ready}/{len(self.allocations)} servers ready ({elapsed}s elapsed)")
            
            if ports_ready == len(self.allocations):
                logger.info(f"All {len(self.allocations)} servers are ready after {elapsed}s")
                break
        else:
            ports_ready = sum(1 for i in range(len(self.allocations)) 
                             if check_port(self.config.grpc_port_base + i))
            alive_count = sum(1 for p in processes if p.is_alive())
            logger.error(f"Only {ports_ready}/{len(self.allocations)} servers ready after {max_wait}s")
            logger.error(f"{alive_count}/{len(self.allocations)} processes still alive")
            for i, p in enumerate(processes):
                logger.error(f"Process {i} (PID {p.pid}): alive={p.is_alive()}, exitcode={p.exitcode}")
            if ports_ready == 0 and alive_count == len(processes):
                raise RuntimeError(
                    f"Servers are running but ports not accessible after {max_wait}s. "
                    f"Model loading may be taking longer than expected. "
                    f"Try using a single model first to test, or check GPU memory."
                )
            elif ports_ready == 0:
                raise RuntimeError(
                    f"All server processes failed. Check logs above for error messages."
                )

        self._server_processes = processes

    async def _create_clients(self):
        for i, allocation in enumerate(self.allocations[:-1]):
            port = self.config.grpc_port_base + i
            client = AsyncLLMClient("localhost", port)
            await client.connect()
            self.chorus_clients.append(client)

        judge_port = self.config.grpc_port_base + len(self.allocations) - 1
        self.judge_client = AsyncLLMClient("localhost", judge_port)
        await self.judge_client.connect()

    async def generate(self, prompt: str) -> ChorusResult:
        start_time = time.time()

        logger.info(f"Sending prompt to {len(self.chorus_clients)} models")

        tasks = [
            client.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                return_logprobs=True,
            )
            for client in self.chorus_clients
        ]

        parallel_start = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        parallel_time = (time.time() - parallel_start) * 1000

        valid_responses = [
            r for r in responses if isinstance(r, GenerationResult)
        ]

        if len(valid_responses) == 0:
            raise RuntimeError("All chorus models failed to generate")

        logger.info(f"Got {len(valid_responses)} valid responses")

        final_answer, reasoning = await self.judge_aggregator.aggregate(
            prompt=prompt,
            responses=valid_responses,
        )

        total_time = (time.time() - start_time) * 1000

        return ChorusResult(
            final_answer=final_answer,
            individual_responses=valid_responses,
            judge_reasoning=reasoning,
            total_latency_ms=total_time,
            parallel_latency_ms=parallel_time,
        )

    async def shutdown(self):
        logger.info("Shutting down VotingChorus...")

        for client in self.chorus_clients:
            await client.close()

        if self.judge_client:
            await self.judge_client.close()

        for process in self._server_processes:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        logger.info("Shutdown complete")

