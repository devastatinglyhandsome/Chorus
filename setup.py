# Setup script with automatic protobuf generation

from setuptools import setup
from setuptools.command.build_py import build_py
from pathlib import Path
import subprocess
import sys


class BuildPyCommand(build_py):
    def run(self):
        self.generate_protobuf()
        super().run()

    def generate_protobuf(self):
        proto_dir = Path(__file__).parent / "chorus" / "server" / "protos"
        proto_file = proto_dir / "inference.proto"

        if not proto_file.exists():
            return

        try:
            subprocess.run(
                [
                    sys.executable, "-m", "grpc_tools.protoc",
                    "-I.", "--python_out=.", "--grpc_python_out=.",
                    "inference.proto"
                ],
                cwd=proto_dir,
                check=True,
            )
            
            grpc_file = proto_dir / "inference_pb2_grpc.py"
            if grpc_file.exists():
                with open(grpc_file, 'r') as f:
                    content = f.read()
                
                if 'import inference_pb2 as inference__pb2' in content:
                    content = content.replace(
                        'import inference_pb2 as inference__pb2',
                        'from . import inference_pb2 as inference__pb2'
                    )
                    
                    with open(grpc_file, 'w') as f:
                        f.write(content)
            
            print("Generated protobuf files successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not generate protobuf files: {e}")


setup(
    packages=["chorus", "chorus.core", "chorus.server", "chorus.server.protos", "chorus.client", "chorus.aggregation", "chorus.benchmarks", "chorus.dashboard"],
    cmdclass={"build_py": BuildPyCommand},
)
