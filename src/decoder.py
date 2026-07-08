from enum import Enum
from typing import Any
import numpy as np
import json


class State(Enum):
    EXPECTING_OPEN_BRACE = "expecting_open_brace"
    EXPECTING_NAME = "expecting_name"
    EXPECTING_COLON = "expecting_colon"
    INSIDE_FUNCTION_NAME = "inside_function_name"
    EXPECTING_COMMA = "expecting_comma"
    EXPECTING_PARAMETERS = "expecting_parameters"
    EXPECTING_OPEN_PARAMETER_BRACE = "expecting_open_parameter_brace"
    EXPECTING_PARAMETER_KEY = "expecting_parameter_key"
    INSIDE_PARAMETER_VALUE_NUMBER = "inside_parameter_value_number"
    INSIDE_PARAMETER_VALUE_STRING = "inside_parameter_value_string"
    INSIDE_PARAMETER_VALUE_BOOLEAN = "inside_parameter_value_boolean"
    EXPECTING_CLOSING_PARAMETER_BRACE = "expecting_closing_parameter_brace"
    EXPECTING_CLOSING_BRACE = "expecting_closing_brace"
    DONE = "done"


class Decoder:
    def __init__(
        self,
        model: Any,
        vocab: dict[int, str],
        function_definitions: list[Any],
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.function_definitions = function_definitions
        self.current_state: State = State.EXPECTING_OPEN_BRACE
        self.valid_tokens_per_state: dict[State, set[int]] = (
            self._precompute_valid_tokens()
        )
        self.generate_within_state: str = ""
        self.position_within_state: int = 0
        self.encoded_name: list[int] = self.model.encode('"name"').tolist()[0]
        self.encoded_parameters: list[int] = self.model.encode(
            '"parameters"'
        ).tolist()[0]
        self.encoded_function_names: dict[str, list[int]] = {
            func.name: self.model.encode('"' + func.name + '"').tolist()[0]
            for func in self.function_definitions
        }
        self.encoded_parameter_keys: dict[str, dict[str, list[int]]] = {}
        for func in self.function_definitions:
            self.encoded_parameter_keys[func.name] = {
                key: self.model.encode('"' + key + '"').tolist()[0]
                for key in func.parameters.keys()
            }
        self.used_parameter_keys: list[str] = []
        self.current_parameter_key: str = ""
        self.chosen_function_name: str = ""
        self.generated_tokens_within_state: list[int] = []

        self.token_without_quote: set[int] = {
            id
            for id, text in self.vocab.items()
            if '"' not in text and "\n" not in text
        }
        self.quote_tokens: set[int] = set(self._get_tokens_for_string('"'))
        self.bool_token_map: dict[int, bool] = {
            self.model.encode("true").tolist()[0][0]: True,
            self.model.encode("false").tolist()[0][0]: False,
        }

        self.prompt: str = ""

        self.current_request: str = ""

    def _get_tokens_for_string(self, target: str) -> set[int]:
        result: set[int] = set()
        for token_id, token_str in self.vocab.items():
            if token_str == target:
                result.add(token_id)
        return result

    def _precompute_valid_tokens(self) -> dict[State, set[int]]:
        result: dict[State, set[int]] = {}
        result[State.EXPECTING_OPEN_BRACE] = self._get_tokens_for_string("{")
        result[State.EXPECTING_COLON] = self._get_tokens_for_string(":")
        result[State.EXPECTING_COMMA] = self._get_tokens_for_string(",")
        result[State.EXPECTING_OPEN_PARAMETER_BRACE] = (
            self._get_tokens_for_string("{")
        )
        result[State.EXPECTING_CLOSING_PARAMETER_BRACE] = (
            self._get_tokens_for_string("}")
        )
        result[State.EXPECTING_CLOSING_BRACE] = self._get_tokens_for_string(
            "}"
        )
        result[State.INSIDE_FUNCTION_NAME] = self._inside_function_name()
        result[State.INSIDE_PARAMETER_VALUE_NUMBER] = self._get_number_tokens()
        result[State.INSIDE_PARAMETER_VALUE_BOOLEAN] = (
            self._get_boolean_tokens()
        )
        result[State.INSIDE_PARAMETER_VALUE_STRING] = self._get_string_tokens()
        result[State.EXPECTING_PARAMETER_KEY] = (
            self._expecting_paramerter_key()
        )
        return result

    def _compute_tier2_tokens(self, target: list[int]) -> set[int]:
        result: set[int] = set()
        if self.position_within_state >= len(target):
            return result
        else:
            result.add(target[self.position_within_state])
        return result

    def _inside_function_name(self) -> set[int]:
        result: set[int] = set()
        for func in self.function_definitions:
            func_name = self.model.encode('"' + func.name + '"').tolist()[0]
            for id in func_name:
                result.add(id)
        return result

    def _expecting_paramerter_key(self) -> set[int]:
        result: set[int] = set()
        for func in self.function_definitions:
            params = func.parameters.keys()
            for k in params:
                func_params = self.model.encode('"' + k + '"').tolist()[0]
                for key_params in func_params:
                    result.add(key_params)
        return result

    def _get_number_tokens(self) -> set[int]:
        result: set[int] = set()
        for token_id, token_str in self.vocab.items():
            if token_str != "" and all(
                c.isdigit() or c == "." or c == "-" for c in token_str
            ):
                result.add(token_id)
        return result

    def _get_boolean_tokens(self) -> set[int]:
        result: set[int] = set()
        for token_id, token_str in self.vocab.items():
            if token_str == "true" or token_str == "false":
                result.add(token_id)
        return result

    def _get_string_tokens(self) -> set[int]:
        result: set[int] = set()
        for token_id, token_str in self.vocab.items():
            if (
                '"' not in token_str or token_str == '"'
            ) and "\n" not in token_str:
                result.add(token_id)
        return result

    def _valid_parameter_key_tokens(self) -> set[int]:
        result: set[int] = set()
        if self.chosen_function_name not in self.encoded_parameter_keys:
            return result

        prefix_length = len(self.generated_tokens_within_state)
        for key_name, encoded_key in self.encoded_parameter_keys[
            self.chosen_function_name
        ].items():
            if key_name in self.used_parameter_keys:
                continue
            if (
                encoded_key[:prefix_length]
                == self.generated_tokens_within_state
            ):
                if prefix_length < len(encoded_key):
                    result.add(encoded_key[prefix_length])
        return result

    def _valid_value_terminators(self) -> set[int]:
        result: set[int] = set()
        for func in self.function_definitions:
            if func.name == self.chosen_function_name:
                total_keys = len(func.parameters)
                if len(self.used_parameter_keys) < total_keys:
                    return self._get_tokens_for_string(",")
                else:
                    return self._get_tokens_for_string("}")
        return result

    def _mask_logits(self, logits: Any, valid_tokens: set[int]) -> Any:
        arr = np.array(logits)
        cpy = arr.copy()
        arr[:] = -float("inf")
        arr[list(valid_tokens)] = cpy[list(valid_tokens)]
        return arr

    def _valid_function_name_tokens(self) -> set[int]:
        result: set[int] = set()
        prefix_length = len(self.generated_tokens_within_state)
        for tokens in self.encoded_function_names.values():
            if tokens[:prefix_length] == self.generated_tokens_within_state:
                result.add(tokens[len(self.generated_tokens_within_state)])
        return result

    def _at_value_boundary(self, content_text: str) -> bool:
        found = False
        start = 0
        while True:
            i = self.prompt.find(content_text, start)
            if i == -1:
                break
            found = True
            end = i + len(content_text)
            if end >= len(self.prompt):
                pass
            elif self.prompt[end] == '"':
                pass
            elif self.prompt[end] == "'" and (
                end + 1 >= len(self.prompt)
                or self.prompt[end + 1] in " ,.?!;:"
            ):

                pass
            else:
                return False
            start = i + 1
        return found

    def _string_valid_parameter(self) -> set[int]:
        if self.quote_tokens.isdisjoint(self.generated_tokens_within_state):

            return self.quote_tokens
        content_tokens = self.generated_tokens_within_state[1:]
        if content_tokens:
            content_text = self.model.decode(content_tokens)

            stripped = content_text.lstrip(" ")
            if stripped and self._at_value_boundary(stripped):

                return self.quote_tokens
        return self.token_without_quote | self.quote_tokens

    def _converted_int_para(self, decoded_para: str) -> int | float | None:
        for func in self.function_definitions:
            if func.name == self.chosen_function_name:
                param_type = func.parameters[self.current_parameter_key].type
                if param_type == "integer":
                    converted: int | float = int(decoded_para)
                    return converted
                elif param_type == "number":
                    converted = float(decoded_para)
                    return converted
        return None

    def _debug(self) -> None:
        print(self.model.encode("-").tolist()[0])
        print(len(self._get_tokens_for_string("-")))

    def _number_valid_parameter(self) -> set[int]:
        if self.generated_tokens_within_state:
            return (
                self.valid_tokens_per_state[
                    State.INSIDE_PARAMETER_VALUE_NUMBER
                ]
                | self._valid_value_terminators()
            )

        target_index = len(self.used_parameter_keys) - 1
        signs_found: list[str] = []
        i = 0
        while i < len(self.current_request):
            if (
                self.current_request[i] == "-"
                and i + 1 < len(self.current_request)
                and self.current_request[i + 1].isdigit()
            ):
                signs_found.append("-")
                i += 1
                while (
                    i < len(self.current_request)
                    and self.current_request[i].isdigit()
                ):
                    i += 1
            elif self.current_request[i].isdigit() and (
                i == 0 or self.current_request[i - 1] != "-"
            ):
                signs_found.append("+")
                i += 1
                while (
                    i < len(self.current_request)
                    and self.current_request[i].isdigit()
                ):
                    i += 1
            else:
                i += 1

        if (
            target_index < len(signs_found)
            and signs_found[target_index] == "-"
        ):
            return self._get_tokens_for_string("-")
        return (
            self.valid_tokens_per_state[State.INSIDE_PARAMETER_VALUE_NUMBER]
            | self._valid_value_terminators()
        )

    def _get_valid_tokens(
        self, fixed_sequence_states: dict[State, tuple[list[int], State]]
    ) -> set[int]:
        MAX_NUMBER_TOKENS = 10
        MAX_STRING_TOKENS = 25
        if self.current_state in fixed_sequence_states:
            target_seq, _ = fixed_sequence_states[self.current_state]
            valid_tokens = self._compute_tier2_tokens(target_seq)
        elif self.current_state == State.INSIDE_FUNCTION_NAME:
            valid_tokens = self._valid_function_name_tokens()
        elif self.current_state == State.EXPECTING_PARAMETER_KEY:
            valid_tokens = self._valid_parameter_key_tokens()
        elif self.current_state == State.INSIDE_PARAMETER_VALUE_NUMBER:
            if len(self.generated_tokens_within_state) >= MAX_NUMBER_TOKENS:
                valid_tokens = self._valid_value_terminators()
            else:
                valid_tokens = self._number_valid_parameter()
        elif self.current_state == State.INSIDE_PARAMETER_VALUE_STRING:
            if len(self.generated_tokens_within_state) >= MAX_STRING_TOKENS:
                valid_tokens = self.quote_tokens
            else:
                valid_tokens = self._string_valid_parameter()
        else:
            valid_tokens = self.valid_tokens_per_state[self.current_state]
        return valid_tokens

    def _change_state(
        self,
        fixed_sequence_states: dict[State, tuple[list[int], State]],
        previous_state: State | None,
        highest_token_score: int,
        result: dict[str, Any],
        immediate_transitions: dict[State, State],
        state_at_start: State,
    ) -> State:
        if self.current_state in fixed_sequence_states:
            target_seq, next_state = fixed_sequence_states[self.current_state]
            self.position_within_state += 1
            if self.position_within_state >= len(target_seq):
                self.position_within_state = 0
                self.current_state = next_state

        elif self.current_state == State.EXPECTING_COLON:
            if previous_state == State.EXPECTING_NAME:
                self.current_state = State.INSIDE_FUNCTION_NAME
            elif previous_state == State.EXPECTING_PARAMETERS:
                self.current_state = State.EXPECTING_OPEN_PARAMETER_BRACE
            elif previous_state == State.EXPECTING_PARAMETER_KEY:
                for func in self.function_definitions:
                    if func.name == self.chosen_function_name:
                        param_type = func.parameters[
                            self.current_parameter_key
                        ].type
                        if param_type == "number" or param_type == "integer":
                            self.current_state = (
                                State.INSIDE_PARAMETER_VALUE_NUMBER
                            )
                        elif param_type == "string":
                            self.current_state = (
                                State.INSIDE_PARAMETER_VALUE_STRING
                            )
                        elif param_type == "boolean":
                            self.current_state = (
                                State.INSIDE_PARAMETER_VALUE_BOOLEAN
                            )

        elif self.current_state == State.EXPECTING_COMMA:
            if previous_state == State.INSIDE_FUNCTION_NAME:
                self.current_state = State.EXPECTING_PARAMETERS
            elif previous_state in {
                State.INSIDE_PARAMETER_VALUE_NUMBER,
                State.INSIDE_PARAMETER_VALUE_STRING,
                State.INSIDE_PARAMETER_VALUE_BOOLEAN,
            }:
                self.current_state = State.EXPECTING_PARAMETER_KEY

        elif self.current_state == State.INSIDE_FUNCTION_NAME:
            self.generated_tokens_within_state.append(highest_token_score)
            if (
                self.generated_tokens_within_state
                in self.encoded_function_names.values()
            ):
                for name, tokens in self.encoded_function_names.items():
                    if tokens == self.generated_tokens_within_state:
                        result["name"] = name
                        self.chosen_function_name = name
                self.generated_tokens_within_state = []
                self.current_state = State.EXPECTING_COMMA

        elif self.current_state == State.EXPECTING_OPEN_PARAMETER_BRACE:
            for func in self.function_definitions:
                if func.name == self.chosen_function_name:
                    if len(func.parameters) == 0:
                        self.current_state = (
                            State.EXPECTING_CLOSING_PARAMETER_BRACE
                        )
                    else:
                        self.current_state = State.EXPECTING_PARAMETER_KEY

        elif self.current_state == State.EXPECTING_PARAMETER_KEY:
            self.generated_tokens_within_state.append(highest_token_score)
            keys_dict = self.encoded_parameter_keys[self.chosen_function_name]
            if self.generated_tokens_within_state in keys_dict.values():
                for key_name, tokens in keys_dict.items():
                    if tokens == self.generated_tokens_within_state:
                        self.current_parameter_key = key_name
                        self.used_parameter_keys.append(key_name)
                self.generated_tokens_within_state = []
                self.current_state = State.EXPECTING_COLON

        elif self.current_state == State.INSIDE_PARAMETER_VALUE_NUMBER:
            token_str = self.vocab[highest_token_score]
            if token_str == ",":
                decoded_para = self.model.decode(
                    self.generated_tokens_within_state
                )
                converted = self._converted_int_para(decoded_para)
                result["parameters"][self.current_parameter_key] = converted
                self.generated_tokens_within_state = []
                self.current_state = State.EXPECTING_PARAMETER_KEY
            elif token_str == "}":
                decoded_para = self.model.decode(
                    self.generated_tokens_within_state
                )
                converted = self._converted_int_para(decoded_para)
                result["parameters"][self.current_parameter_key] = converted
                self.generated_tokens_within_state = []
                self.current_state = State.EXPECTING_CLOSING_BRACE
            else:

                self.generated_tokens_within_state.append(highest_token_score)

        elif self.current_state == State.INSIDE_PARAMETER_VALUE_STRING:
            if self.quote_tokens.isdisjoint(
                self.generated_tokens_within_state
            ):

                self.generated_tokens_within_state.append(highest_token_score)
            elif highest_token_score in self.quote_tokens:

                decoded = self.model.decode(self.generated_tokens_within_state)

                value = decoded.lstrip('"').lstrip(" ")

                try:
                    new_value = json.loads('"' + value + '"')
                except json.JSONDecodeError:

                    new_value = value
                result["parameters"][self.current_parameter_key] = new_value
                self.generated_tokens_within_state = []
                for func in self.function_definitions:
                    if func.name == self.chosen_function_name:
                        if len(self.used_parameter_keys) < len(
                            func.parameters
                        ):
                            self.current_state = State.EXPECTING_COMMA
                        else:
                            self.current_state = State.EXPECTING_CLOSING_BRACE
            else:

                self.generated_tokens_within_state.append(highest_token_score)

        elif self.current_state == State.INSIDE_PARAMETER_VALUE_BOOLEAN:
            value = self.bool_token_map[highest_token_score]
            result["parameters"][self.current_parameter_key] = value
            for func in self.function_definitions:
                if func.name == self.chosen_function_name:
                    if len(self.used_parameter_keys) < len(func.parameters):
                        self.current_state = State.EXPECTING_COMMA
                    else:
                        self.current_state = State.EXPECTING_CLOSING_BRACE

        elif self.current_state in immediate_transitions:
            self.current_state = immediate_transitions[self.current_state]

        previous_state = state_at_start
        return previous_state

    def generate(self, prompt: str) -> dict[str, Any]:
        self.current_state = State.EXPECTING_OPEN_BRACE
        self.used_parameter_keys = []
        self.generated_tokens_within_state = []
        self.position_within_state = 0
        self.chosen_function_name = ""
        self.current_parameter_key = ""

        self.prompt = prompt

        marker = 'Request: "'
        idx = self.prompt.rfind(marker)
        if idx != -1:
            start = idx + len(marker)
            end = self.prompt.find('"', start)
            self.current_request = (
                self.prompt[start:end] if end != -1 else self.prompt[start:]
            )
        else:
            self.current_request = self.prompt
        immediate_transitions: dict[State, State] = {
            State.EXPECTING_OPEN_BRACE: State.EXPECTING_NAME,
            State.EXPECTING_CLOSING_PARAMETER_BRACE: (
                State.EXPECTING_CLOSING_BRACE
            ),
            State.EXPECTING_CLOSING_BRACE: State.DONE,
        }
        fixed_sequence_states: dict[State, tuple[list[int], State]] = {
            State.EXPECTING_NAME: (self.encoded_name, State.EXPECTING_COLON),
            State.EXPECTING_PARAMETERS: (
                self.encoded_parameters,
                State.EXPECTING_COLON,
            ),
        }
        input_ids = self.model.encode(prompt).tolist()[0]
        result: dict[str, Any] = {
            "prompt": prompt,
            "name": "",
            "parameters": {},
        }
        previous_state: State | None = None
        while self.current_state != State.DONE:
            state_at_start = self.current_state

            valid_tokens = self._get_valid_tokens(fixed_sequence_states)

            if len(valid_tokens) == 1:
                highest_token_score = next(iter(valid_tokens))
            else:
                logits = self.model.get_logits_from_input_ids(input_ids)
                clean_tokens = self._mask_logits(logits, valid_tokens)
                highest_token_score = int(np.argmax(clean_tokens))
            input_ids.append(highest_token_score)

            previous_state = self._change_state(
                fixed_sequence_states,
                previous_state,
                highest_token_score,
                result,
                immediate_transitions,
                state_at_start,
            )

        return result
