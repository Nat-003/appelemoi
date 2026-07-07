
from src.parser import get_function_calling, get_function_definition, vocab_loader
from src.promptbuilder import build_prompt
import argparse
from llm_sdk import Small_LLM_Model
from src.decoder import Decoder
from src.output import generate_output
import json
import time


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
        parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
        parser.add_argument(
                            "--output",
                            type=str,
                            default="data/output/function_calling_results.json",
                            )
        args = parser.parse_args()         
        args.output
        function_definition =  get_function_definition(args.functions_definition)
        function_calling = get_function_calling(args.input )
        if function_definition is None or function_calling is None:
            return
        model = Small_LLM_Model()
        vocab = vocab_loader(model)
        decoder = Decoder(model, vocab, function_definition)
        data = []
        total_start = time.perf_counter()
        for item in function_calling:
            start = time.perf_counter()
            prompt_f = build_prompt(function_definition, item)
            result = decoder.generate(prompt_f)
            result["prompt"] = item.prompt
            data.append(result)
            elapsed = time.perf_counter() - start
            print(f"{elapsed:6.2f}s  {item.prompt}")
        total = time.perf_counter() - total_start
        print(f"\nTotal: {total:.2f}s for {len(data)} prompts "
            f"({total / len(data):.2f}s avg)")
        generate_output(data, args.output)
        result = decoder._string_valid_parameter()
    except FileNotFoundError:
        print("Invalide fiel path")
if __name__ == "__main__":
    main()


