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

import sys
import json
import config
from worker import BackTranslator, NLInferencer
from utils import Util, StatUtil

def backtrans_nli(result_path: str):
    with open(result_path) as fp:
        data = json.load(fp)
    
    btrans = BackTranslator(config.BACKTRANS_MODEL_PATH, config.BACKTRANS_GPUS)
    nli = NLInferencer(config.NLI_MODEL, config.NLI_API_BASE_URL, config.NLI_API_KEY)
    
    for i, d in enumerate(data):
        print(f'Processing {i} of {len(data)}')
        verified = d['verified']
        backtranslated = btrans.batch_generate(verified, config.BACKTRANS_SAMPLING_PARAMS)
        nli_outputs = [nli.generate(d['informal_statement'], gen, config.NLI_SAMPLING_PARAMS)
            for gen in backtranslated]
        translated = []
        for v, n in zip(verified, nli_outputs):
            if Util.extract_bold_text(n) == 'same':
                translated.append(v)
        d['translated'] = translated
        print(f'Finished {i} of {len(data)}, got {len(translated)} statements.')
    
    with open(result_path, 'w') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    StatUtil.get_translated_stat(result_path)

if __name__ == '__main__':
    backtrans_nli(sys.argv[1])