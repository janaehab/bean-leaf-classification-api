"""Export a labeled validation image for manual API smoke tests / latency benchmarking."""
import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="sample_leaf.png")
    args = parser.parse_args()
    ds = load_dataset("AI-Lab-Makerere/beans", split="validation")
    if not 0 <= args.index < len(ds):
        parser.error("--index outside validation split")
    row = ds[args.index]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    row["image"].save(args.output)
    print(f"Saved {args.output}; actual label: {ds.features['labels'].int2str(row['labels'])}")


if __name__ == "__main__":
    main()
