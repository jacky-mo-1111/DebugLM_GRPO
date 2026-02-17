import json
import random
from pathlib import Path


def main():
    random.seed(42)
    base = Path(__file__).parent
    src = base / "routing_dataset.json"
    train_out = base / "train.json"
    val_out = base / "val.json"

    with open(src, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    split_idx = int(len(data) * 0.95)
    train, val = data[:split_idx], data[split_idx:]

    with open(train_out, "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(val_out, "w") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Total {len(data)}, train {len(train)}, val {len(val)}")


if __name__ == "__main__":
    main()










