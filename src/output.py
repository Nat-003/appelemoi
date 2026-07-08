import os
import json


def generate_output(
    data: list,
    path: str = "data/output/function_calling_results.json",
) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        print(f"Could not write output file: {exc}")
