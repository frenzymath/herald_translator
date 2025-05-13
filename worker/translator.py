# Copyright 2025 Frenzy Math
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

from vllm import LLM, SamplingParams
from utils import Util

class Translator:
    def __init__(self, model_path: str, gpus=1):
        """
        """
        self.model_path = model_path
        self.model = None
        self.gpus = gpus

    def _init_model(self):
        if self.model is None:
            self.model = LLM(
                model=self.model_path,
                tensor_parallel_size=self.gpus,
                trust_remote_code=True,
                dtype='bfloat16',
            )

    def release_model(self):
        self.model = None

    def get_query(self, informal_name: str, informal_statement: str):
        template = "Please translate the natural language statement to Lean4 code with the header\n**Name**\n{informal_name}\n**Informal statement**\n{informal_statement}\n"
        msgs = [
            {'role': 'system', 'content': 'You are an expert at Lean 4 and Mathematics.'},
            {'role': 'user', 'content': template.format(
                informal_name=informal_name,
                informal_statement=informal_statement)}
        ]
        return Util.chat_template_to_prompt(msgs, 'deepseek')
    
    def generate(self, informal_name: str, informal_statement: str, sampling_params: dict) -> list[str]:
        if self.model is None:
            self._init_model()
        prompt = self.get_query(informal_name, informal_statement)
        output = self.model.generate( # type: ignore
            prompt, sampling_params=SamplingParams(**sampling_params)) 
        return [o.text for o in output[0].outputs]

    def batch_generate(self, items: list[dict], sampling_params: dict) -> list[list[str]]:
        """
        """
        if self.model is None:
            self._init_model()
        prompts = [self.get_query(i['id'], i['informal_statement']) for i in items]
        outputs = self.model.generate(prompts, sampling_params=SamplingParams(**sampling_params)) # type: ignore
        return [[o.text for o in output.outputs] for output in outputs]
    
    def _build_sampling_param(self, sampling_params: dict) -> SamplingParams:
        return SamplingParams(**sampling_params)