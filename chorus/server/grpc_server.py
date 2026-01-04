# gRPC server wrapping vLLM models for inference

import grpc
import logging
from concurrent import futures
from typing import Optional
import time

from vllm import LLM, SamplingParams
from chorus.server.protos import inference_pb2, inference_pb2_grpc
from chorus.core.gpu_manager import ModelAllocation, QuantizationStrategy

logger = logging.getLogger(__name__)


class LLMInferenceServicer(inference_pb2_grpc.LLMInferenceServicer):
    def __init__(self, allocation: ModelAllocation):
        self.allocation = allocation
        self.model: Optional[LLM] = None
        self.model_name = allocation.model_name

    def load_model(self):
        logger.info(f"Loading model {self.model_name} on GPU {self.allocation.gpu_id}")

        quantization_map = {
            QuantizationStrategy.FP16: None,
            QuantizationStrategy.AWQ: "awq",
            QuantizationStrategy.GPTQ: "gptq",
            QuantizationStrategy.INT4: "awq",
            QuantizationStrategy.INT8: "awq",
        }

        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.allocation.memory_fraction,
            quantization=quantization_map[self.allocation.quantization],
            trust_remote_code=True,
        )

        logger.info(f"Model {self.model_name} loaded successfully")

    def Generate(self, request, context):
        if self.model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model not loaded")

        start_time = time.time()

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=list(request.stop_sequences) if request.stop_sequences else None,
            logprobs=1 if request.return_logprobs else None,
        )

        outputs = self.model.generate([request.prompt], sampling_params)
        output = outputs[0]
        generation_time = (time.time() - start_time) * 1000

        logprobs = []
        if request.return_logprobs and output.outputs[0].logprobs:
            all_logprobs = [
                list(token_logprobs.values())[0].logprob
                for token_logprobs in output.outputs[0].logprobs
            ]
            logprobs = all_logprobs

        response = inference_pb2.GenerateResponse(
            text=output.outputs[0].text,
            logprobs=logprobs,
            tokens_generated=len(output.outputs[0].token_ids),
            generation_time_ms=generation_time,
            model_name=self.model_name,
        )

        return response

    def HealthCheck(self, request, context):
        import torch

        if self.model is None:
            status = inference_pb2.HealthCheckResponse.NOT_SERVING
        else:
            status = inference_pb2.HealthCheckResponse.SERVING

        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(self.allocation.gpu_id) / (1024**3)

        return inference_pb2.HealthCheckResponse(
            status=status,
            model_name=self.model_name,
            gpu_memory_used_gb=gpu_memory,
        )


def serve(allocation: ModelAllocation, port: int):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(allocation.gpu_id)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = LLMInferenceServicer(allocation)

    servicer.load_model()

    inference_pb2_grpc.add_LLMInferenceServicer_to_server(servicer, server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f"gRPC server started on port {port} for model {allocation.model_name}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


def main():
    import argparse
    from chorus.core.gpu_manager import ModelAllocation, QuantizationStrategy

    parser = argparse.ArgumentParser(description="Start gRPC model server")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU device ID")
    parser.add_argument("--memory-fraction", type=float, default=0.9, help="GPU memory fraction")
    parser.add_argument("--quantization", default="fp16", help="Quantization strategy")
    parser.add_argument("--port", type=int, required=True, help="gRPC port")

    args = parser.parse_args()

    allocation = ModelAllocation(
        model_name=args.model,
        gpu_id=args.gpu_id,
        memory_fraction=args.memory_fraction,
        quantization=QuantizationStrategy(args.quantization),
    )

    serve(allocation, args.port)


if __name__ == "__main__":
    main()

