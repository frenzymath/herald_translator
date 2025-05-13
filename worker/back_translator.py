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

class BackTranslator:
    def __init__(self, model_path: str, gpus=1):
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

    def get_query(self, formal_statement: str) -> str:
        prompt = f'[UNUSED_TOKEN_146]user\nConvert the formal statement into natural language:\n```lean\n{formal_statement}\n```[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        return prompt
    
    def generate(self, formal_statement: str, sampling_params: dict) -> str:
        if self.model is None:
            self._init_model()
        prompt = self.get_query(formal_statement)
        output = self.model.generate(prompt, sampling_params=SamplingParams(**sampling_params)) # type: ignore
        return output[0].outputs[0].text

    def batch_generate(self, items: list[str], sampling_params: dict) -> list[str]:
        """
        """
        if self.model is None:
            self._init_model()
        prompts = [self.get_query(i) for i in items]
        outputs = self.model.generate(prompts, sampling_params=SamplingParams(**sampling_params)) # type: ignore
        return [output.outputs[0].text for output in outputs]
    
    def _build_sampling_param(self, sampling_params: dict) -> SamplingParams:
        return SamplingParams(**sampling_params)