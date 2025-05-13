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
from worker import Translator, Verifier
from utils import Util, StatUtil

def translate_verify(data_path: str, result_path: str):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    
    trans = Translator(config.TRANS_MODEL_PATH, config.TRANS_GPUS)
    ver = Verifier(config.LEAN_HEADER, config.LEAN_TEST_PATH, config.LAKE_BIN)

    for i, d in enumerate(data):
        print(f'Processing {i} of {len(data)}')
        generated = trans.generate(d['id'], d['informal_statement'], config.TRANS_SAMPLING_PARAMS)
        generated = [Util.remove_informal_prefix(g) for g in generated]
        verified = ver.batch_verify_item(generated)
        d['verified'] = verified
        print(f'Finished {i} of {len(data)}, got {len(verified)} statements')
    
    with open(result_path, 'w') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    StatUtil.get_verified_stat(result_path)

if __name__ == '__main__':
    translate_verify(sys.argv[1], sys.argv[2])