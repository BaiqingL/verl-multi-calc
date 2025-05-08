# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CalculatorTool(BaseTool):
    """A tool for performing basic arithmetic calculations.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a calculation.
    - `execute`: execute the calculation.
    - `calc_reward`: calculate the reward based on calculation accuracy.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, expected_result: Optional[float] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "expression": "",
            "result": None,
            "expected_result": expected_result,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        try:
            _parameters = json.loads(parameters)
        except json.JSONDecodeError:
            _parameters = {}
        
        if isinstance(_parameters, dict):
            expression = _parameters.get("expression", "")
            if not isinstance(expression, str):
                expression = str(expression)
        else:
            expression = ""
            
        try:
            # Safely evaluate the expression
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
            if not isinstance(result, (int, float)):
                result = float(result)
        except Exception as e:
            result = None
            logger.warning(f"Error evaluating expression: {e}")
            
        self._instance_dict[instance_id]["expression"] = expression
        self._instance_dict[instance_id]["result"] = result
        
        reward = await self.calc_reward(instance_id)
        # penalty for incorrect calculations
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        
        return f"Expression: {expression}, Result: {result}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        result = self._instance_dict[instance_id]["result"]
        expected = self._instance_dict[instance_id]["expected_result"]
        
        if result is None or expected is None:
            return 0.0
            
        # Calculate reward based on how close the result is to expected
        if abs(result - expected) < 1e-10:  # For exact matches
            return 1.0
        elif abs(result - expected) < 0.01:  # For very close matches
            return 0.8
        elif abs(result - expected) < 0.1:  # For close matches
            return 0.5
        else:
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
