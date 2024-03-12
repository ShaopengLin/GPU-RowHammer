import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GPU Timing Channel Main",
        description="Facade Runner that executes different CUDA timing channel tasks",
    )
    args = parser.parse_args()
