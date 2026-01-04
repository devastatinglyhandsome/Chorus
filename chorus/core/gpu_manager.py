# GPU detection, validation, and model allocation management

import torch
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class QuantizationStrategy(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass
class GPUInfo:
    device_id: int
    name: str
    total_memory_gb: float
    compute_capability: Tuple[int, int]


@dataclass
class ModelAllocation:
    model_name: str
    gpu_id: int
    memory_fraction: float
    quantization: QuantizationStrategy


class GPUManager:
    def __init__(self, num_gpus: int, on_insufficient_memory: str = "quantize"):
        self.num_gpus = num_gpus
        self.on_insufficient_memory = on_insufficient_memory
        self.gpus: List[GPUInfo] = []

    def detect_gpus(self) -> List[GPUInfo]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for chorus")

        detected_count = torch.cuda.device_count()
        if detected_count < self.num_gpus:
            raise ValueError(
                f"Requested {self.num_gpus} GPUs but only {detected_count} available. "
                f"Set num_gpus={detected_count} or add more GPUs."
            )

        gpus = []
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                device_id=i,
                name=props.name,
                total_memory_gb=props.total_memory / (1024**3),
                compute_capability=(props.major, props.minor)
            ))

        self.gpus = gpus
        return gpus

    def estimate_model_memory(self, model_name: str, quantization: QuantizationStrategy) -> float:
        if "1b" in model_name.lower() or "1B" in model_name:
            params = 1e9
        elif "3b" in model_name.lower() or "3B" in model_name:
            params = 3e9
        elif "7b" in model_name.lower() or "7B" in model_name:
            params = 7e9
        elif "13b" in model_name.lower() or "13B" in model_name:
            params = 13e9
        else:
            params = 3e9

        bytes_per_param = {
            QuantizationStrategy.FP16: 2.0,
            QuantizationStrategy.INT8: 1.0,
            QuantizationStrategy.INT4: 0.5,
            QuantizationStrategy.AWQ: 0.5,
            QuantizationStrategy.GPTQ: 0.5,
        }

        base_memory = (params * bytes_per_param[quantization]) / (1024**3)
        total_memory = base_memory * 1.3
        return total_memory

    def plan_allocation(self, chorus_models: List[str], judge_model: str) -> List[ModelAllocation]:
        allocations = []
        quantization = QuantizationStrategy.FP16

        chorus_memory = sum(
            self.estimate_model_memory(model, quantization)
            for model in chorus_models
        )
        judge_memory = self.estimate_model_memory(judge_model, quantization)

        gpu0_available = self.gpus[0].total_memory_gb
        gpu1_available = self.gpus[1].total_memory_gb if len(self.gpus) > 1 else 0

        if chorus_memory > gpu0_available:
            if self.on_insufficient_memory == "error":
                raise MemoryError(
                    f"Chorus models require {chorus_memory:.1f}GB but GPU 0 "
                    f"only has {gpu0_available:.1f}GB. "
                    f"Try: on_insufficient_memory='quantize' or use smaller models."
                )
            elif self.on_insufficient_memory == "quantize":
                quantization = QuantizationStrategy.INT4
                chorus_memory = sum(
                    self.estimate_model_memory(model, quantization)
                    for model in chorus_models
                )

                if chorus_memory > gpu0_available:
                    raise MemoryError(
                        f"Even with INT4 quantization, models need {chorus_memory:.1f}GB "
                        f"but GPU 0 has {gpu0_available:.1f}GB. Use smaller/fewer models."
                    )

        memory_per_model = gpu0_available / len(chorus_models)
        for model in chorus_models:
            allocations.append(ModelAllocation(
                model_name=model,
                gpu_id=0,
                memory_fraction=memory_per_model / gpu0_available,
                quantization=quantization
            ))

        judge_gpu = 1 if len(self.gpus) > 1 else 0
        judge_available = gpu1_available if judge_gpu == 1 else gpu0_available

        allocations.append(ModelAllocation(
            model_name=judge_model,
            gpu_id=judge_gpu,
            memory_fraction=min(0.9, judge_memory / judge_available),
            quantization=QuantizationStrategy.FP16
        ))

        return allocations

