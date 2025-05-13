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

# Configure Herald Translator
TRANS_MODEL_PATH = 'FrenzyMath/Herald_translator'
TRANS_GPUS = 4
TRANS_SAMPLING_PARAMS = {
    'max_tokens': 1024,
    'temperature': 0.99,
    'n': 128
}

# Configure Back-translate
BACKTRANS_MODEL_PATH = 'internlm/internlm2-math-plus-7b'
BACKTRANS_GPUS = 4
BACKTRANS_SAMPLING_PARAMS = {
    'temperature': 0.1,
    'max_tokens': 1024,
    'stop': ['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]', '<|im_end|>']
}

# Configure Lean environment
LEAN_HEADER = 'import Mathlib'
LEAN_TEST_PATH = '../leantest'
LAKE_BIN = '~/.elan/bin/lake'

# Configure NLI model
NLI_MODEL = 'deepseek-ai/DeepSeek-V2.5'
NLI_API_BASE_URL = 'https://api.deepseek.com'
NLI_API_KEY = 'sk-xxxxxxxxxxxxxx'
NLI_SAMPLING_PARAMS = {
    'max_tokens': 2048,
    'temperature':0.01,
    'top_p': 0.7,
    'frequency_penalty': 1
}