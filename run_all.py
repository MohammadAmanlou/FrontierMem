from run_demo import main as run_baselines
from run_tiny_llm import main as run_tiny_lm


if __name__ == "__main__":
    run_baselines()
    print("\n" + "=" * 80 + "\n")
    run_tiny_lm()
