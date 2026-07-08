from src.parser import FunctionCalling, FunctionDefinition
import json


def build_prompt(
        function_definitions: list[FunctionDefinition],
        function_calling: FunctionCalling
        ) -> str:
    lines = [
        "You are a function-calling assistant. "
        "Choose the single best function for the request "
        "and extract its arguments exactly as given. "
        "Keep arguments minimal and literal.",
        "",
        "Available functions:",
    ]
    for f in function_definitions:
        params = ", ".join(
            f"{name}: {p.type}" for name, p in f.parameters.items()
        )
        lines.append(f"- {f.name}({params}): {f.description}")

    # one-shot example built from the FIRST real function, so it always
    # references a function that actually exists in this definition set
    first = function_definitions[0]
    example_args = {}
    for name, p in first.parameters.items():
        if p.type == "number":
            example_args[name] = 1
        elif p.type == "boolean":
            example_args[name] = True
        else:
            example_args[name] = "example"
    example_call = {"name": first.name, "parameters": example_args}

    lines += [
        "",
        "Example:",
        f'Request: "Call {first.name} with sample values"',
        f"Call: {json.dumps(example_call)}",
        "",
        f'Request: "{function_calling.prompt}"',
        "Call:",
    ]
    return "\n".join(lines)
