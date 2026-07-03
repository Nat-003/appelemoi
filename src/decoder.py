from enum import Enum
from typing import Any
import numpy as np

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
    def __init__(self, model, vocab, function_definitions):
        self.model = model
        self.vocab = vocab
        self.function_definitions = function_definitions
        self.current_state = State.EXPECTING_OPEN_BRACE
        self.valid_tokens_per_state = self._precompute_valid_tokens()
        self.generate_within_state = ""
        self.position_within_state = 0
        self.encoded_name = self.model.encode('"name"').tolist()[0]
        self.encoded_parameters = self.model.encode('"parameters"').tolist()[0]
        self.encoded_function_names = {
                                        func.name: self.model.encode('"' + func.name + '"').tolist()[0]
                                        for func in self.function_definitions
                                        }
        self.encoded_parameter_keys = {}
        for func in self.function_definitions:
            self.encoded_parameter_keys[func.name] = {
                key: self.model.encode('"' + key + '"').tolist()[0]
                for key in func.parameters.keys()
            }
        self.used_parameter_keys = []
        self.current_parameter_key = "" 
        self.chosen_function_name = ""
        self.generated_tokens_within_state = []
        self.token_without_quote = { id for id, text in self.vocab.items()  if '"' not in text}
        self.quote_tokens = set(self._get_tokens_for_string('"'))
        self.bool_token_map = {
                                self.model.encode("true").tolist()[0][0]: True,
                                self.model.encode("false").tolist()[0][0]: False,
                            }

    def _get_tokens_for_string(self, target: str) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str == target:
                result.add(token_id)
        return result

    
    def _precompute_valid_tokens(self) -> dict[State, set[int]]:
        result = {}
        result[State.EXPECTING_OPEN_BRACE] = self._get_tokens_for_string("{")
        result[State.EXPECTING_COLON] = self._get_tokens_for_string(":")
        result[State.EXPECTING_COMMA] = self._get_tokens_for_string(",")
        result[State.EXPECTING_OPEN_PARAMETER_BRACE] = self._get_tokens_for_string("{")
        result[State.EXPECTING_CLOSING_PARAMETER_BRACE] = self._get_tokens_for_string("}")
        result[State.EXPECTING_CLOSING_BRACE] = self._get_tokens_for_string("}")
        result[State.INSIDE_FUNCTION_NAME] = self._inside_function_name()
        result[State.INSIDE_PARAMETER_VALUE_NUMBER] = self._get_number_tokens()
        result[State.INSIDE_PARAMETER_VALUE_BOOLEAN] = self._get_boolean_tokens()
        result[State.INSIDE_PARAMETER_VALUE_STRING] = self._get_string_tokens()
        result[State.EXPECTING_PARAMETER_KEY] = self._expecting_paramerter_key()
        return result

    def _compute_tier2_tokens(self, target: list[int]) -> set[int]:
        result = set()
        if self.position_within_state >= len(target):
            return result
        else:
            result.add(target[self.position_within_state])
        return result
    
    def _inside_function_name(self) -> set[int]:
        result = set()
        for func in self.function_definitions:
            func_name = self.model.encode('"' + func.name + '"').tolist()[0]
            for id in func_name:
                result.add(id)
        return result

    def _expecting_paramerter_key(self) -> set[int]:
        result = set()
        for func in self.function_definitions:
            params = func.parameters.keys()
            for k in params:
                func_params = self.model.encode('"' + k + '"').tolist()[0]
                for key_params in func_params:
                    result.add(key_params)
        return result

    def _get_number_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str != "" and all(c.isdigit() or c == '.' or  c == '-' for c in token_str):
                result.add(token_id)
        return result

    def _get_boolean_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str == "true" or token_str == "false":
                result.add(token_id)
        return result

    def _get_string_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if '"' not in token_str or token_str == '"':
                result.add(token_id)
        return result

    def _valid_parameter_key_tokens(self) -> set[int]:
        result = set()
        if self.chosen_function_name not in self.encoded_parameter_keys:
            return result
        
        prefix_length = len(self.generated_tokens_within_state)
        for key_name, encoded_key in self.encoded_parameter_keys[self.chosen_function_name].items():
            if key_name in self.used_parameter_keys:
                continue  # skip used keys
            if encoded_key[:prefix_length] == self.generated_tokens_within_state:
                if prefix_length < len(encoded_key):
                    result.add(encoded_key[prefix_length])
        return result

    def _valid_value_terminators(self) -> set[int]:
        result = set()
        for func in self.function_definitions:
            if func.name == self.chosen_function_name:
                total_keys = len(func.parameters)
                if len(self.used_parameter_keys) < total_keys:
                    return self._get_tokens_for_string(",")
                else:
                    return self._get_tokens_for_string("}")
        return result


    def _mask_logits(self ,logits: Any, valid_tokens: set[int]) -> Any:
        arr = np.array(logits)
        cpy = arr.copy()
        arr[:] = -float('inf')
        arr[list(valid_tokens)] = cpy[list(valid_tokens)]
        return arr

    def _valid_function_name_tokens(self) -> set[int]:
        result = set()
        prefix_length = len(self.generated_tokens_within_state)
        for tokens in self.encoded_function_names.values():
            if tokens[:prefix_length] == self.generated_tokens_within_state:
                result.add(tokens[len(self.generated_tokens_within_state)])
        return result

    def _string_valid_parameter(self) -> set[int]:
        quote_tokens = set(self._get_tokens_for_string('"'))
        if quote_tokens.isdisjoint(self.generated_tokens_within_state):
            return quote_tokens
        else:
            return self.token_without_quote | quote_tokens

    # def _debug(self,target):
    #     for token_id, token_str in self.vocab.items():
    #         if token_id == target:
    #             print(f"the word/sub word is {token_str}")

    def _converted_int_para(self, decoded_para: str) -> int | float:
        for func in self.function_definitions:
            if func.name == self.chosen_function_name:
                param_type = func.parameters[self.current_parameter_key].type
                print(param_type)
                if param_type == "integer":
                    converted = int(decoded_para)
                    return converted
                elif param_type == "number":
                    converted = float(decoded_para)
                    return converted

    def generate(self, prompt) -> dict:
        self.current_state = State.EXPECTING_OPEN_BRACE
        self.used_parameter_keys = []
        self.generated_tokens_within_state = []
        self.position_within_state = 0
        self.chosen_function_name = ""
        self.current_parameter_key = ""
        immediate_transitions = {
            State.EXPECTING_OPEN_BRACE: State.EXPECTING_NAME,
            State.EXPECTING_OPEN_PARAMETER_BRACE: State.EXPECTING_PARAMETER_KEY,
            State.EXPECTING_CLOSING_PARAMETER_BRACE: State.EXPECTING_CLOSING_BRACE,
            State.EXPECTING_CLOSING_BRACE: State.DONE,
        }
        fixed_sequence_states = {
            State.EXPECTING_NAME: (self.encoded_name, State.EXPECTING_COLON),
            State.EXPECTING_PARAMETERS: (self.encoded_parameters, State.EXPECTING_COLON),
        }

        input_ids = self.model.encode(prompt).tolist()[0]
        result = {"prompt": prompt, "name": "", "parameters": {}}
        previous_state = None
        MAX_NUMBER_TOKENS = 10
        MAX_STRING_TOKENS = 20
        while self.current_state != State.DONE:
            state_at_start = self.current_state
            # print(self.current_state)

            # --- 1. Get valid tokens for current state ---
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
                    valid_tokens = self.valid_tokens_per_state[State.INSIDE_PARAMETER_VALUE_NUMBER] | self._valid_value_terminators()
            elif self.current_state == State.INSIDE_PARAMETER_VALUE_STRING:
                if len(self.generated_tokens_within_state) >= MAX_STRING_TOKENS:
                    valid_tokens = self.quote_tokens
                else:
                    valid_tokens = self._string_valid_parameter()
            else:
                valid_tokens = self.valid_tokens_per_state[self.current_state]
                            
            
            # --- 2. Get logits, mask, pick token ---
            # --- 2. Pick token: skip the model when only one is legal ---
            if len(valid_tokens) == 1:
                highest_token_score = next(iter(valid_tokens))
            else:
                logits = self.model.get_logits_from_input_ids(input_ids)
                clean_tokens = self._mask_logits(logits, valid_tokens)
                highest_token_score = int(np.argmax(clean_tokens))
            input_ids.append(highest_token_score)
            # print(f"State: {self.current_state.value}, token picked: {highest_token_score}, actual word/subword: {self.model.decode(highest_token_score)}")

            # --- 3. State transitions ---
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
                            param_type = func.parameters[self.current_parameter_key].type
                            # print(f"  param_type: {param_type}")
                            if param_type == "number" or param_type == "integer":
                                self.current_state = State.INSIDE_PARAMETER_VALUE_NUMBER
                            elif param_type == "string":
                                self.current_state = State.INSIDE_PARAMETER_VALUE_STRING
                            elif param_type == "boolean":
                                self.current_state = State.INSIDE_PARAMETER_VALUE_BOOLEAN

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
                if self.generated_tokens_within_state in self.encoded_function_names.values():
                    for name, tokens in self.encoded_function_names.items():
                        if tokens == self.generated_tokens_within_state:
                            result["name"] = name
                            self.chosen_function_name = name
                    self.generated_tokens_within_state = []
                    self.current_state = State.EXPECTING_COMMA

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
                    decoded_para = self.model.decode(self.generated_tokens_within_state)
                    converted = self._converted_int_para(decoded_para)
                    result["parameters"][self.current_parameter_key] = converted
                    # print(f"token_str: {token_str} decoded para: {decoded_para}")
                    self.generated_tokens_within_state = []
                    self.current_state = State.EXPECTING_PARAMETER_KEY
                elif token_str == "}":
                    decoded_para = self.model.decode(self.generated_tokens_within_state)
                    converted = self._converted_int_para(decoded_para)
                    result["parameters"][self.current_parameter_key] = converted
                    # print(f"token_str: {token_str} decoded para: {decoded_para}")
                    self.generated_tokens_within_state = []
                    self.current_state = State.EXPECTING_CLOSING_BRACE
                else:
                    # accumulate
                    self.generated_tokens_within_state.append(highest_token_score)

            elif self.current_state == State.INSIDE_PARAMETER_VALUE_STRING:
                if self.quote_tokens.isdisjoint(self.generated_tokens_within_state):
                    # no quote yet → this is the OPENING quote
                    self.generated_tokens_within_state.append(highest_token_score)
                elif highest_token_score in self.quote_tokens:
                    # already open, another quote → CLOSE → finalize
                    decoded = self.model.decode(self.generated_tokens_within_state)
                    value = decoded.lstrip('"')
                    result["parameters"][self.current_parameter_key] = value
                    self.generated_tokens_within_state = []
                    for func in self.function_definitions:
                        if func.name == self.chosen_function_name:
                            if len(self.used_parameter_keys) < len(func.parameters):
                                self.current_state = State.EXPECTING_COMMA
                            else:
                                self.current_state = State.EXPECTING_CLOSING_BRACE
                else:
                    # already open, ordinary content → accumulate
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

        return result