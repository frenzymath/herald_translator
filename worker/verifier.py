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

import json
import subprocess
import tempfile
import traceback
import concurrent.futures

class Verifier(object):
    def __init__(self, header: str, lean_test_path: str, lake_bin: str, max_repl_threads=20):
        self.lean_test_path = lean_test_path
        self.lake_bin = lake_bin
        self.max_repl_threads = max_repl_threads
        self.header = header

    def verify(self, code_string: str, timeout=600) -> tuple[bool, str]:
        validation = True
        try:
            result = self.verify_one_lean_codestring(code_string, timeout)
            result_json = json.loads(result)
            if result_json.get("messages"):
                for msg in result_json.get("messages"):
                    if msg.get("severity") == "error":
                        validation = False
        except Exception as e:
            print(e)
            validation, result = False, str(e)
        return validation, result

    def verify_one_lean_codestring(self, code_string, timeout=300):
        command = dict(cmd=self.header + '\n' + code_string)
        message_str = json.dumps(command, ensure_ascii=False)
        lean_path = self.lean_test_path
        try:
            with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
                temp_file.write(message_str + "\r\n\r\n")
                temp_file.seek(0)
                outputs = subprocess.run(
                    [self.lake_bin, "exe", 'repl'],
                    stdin=temp_file,
                    capture_output=True,
                    text=True,
                    cwd=lean_path,
                    timeout=timeout,
                    encoding='utf-8'
                )
        except Exception as e:
            print(traceback.format_exc())
            return str(e)
        else:
            return outputs.stdout
        
    def batch_verify(self, code_list: list[str]):
        max_workers = min(len(code_list), self.max_repl_threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.verify, code_list))
        return results

    def batch_verify_item(self, statement_list: list[str]):
        def verify_item(item):
            validation, result = self.verify(item)
            return item if validation else None

        max_workers = min(len(statement_list), self.max_repl_threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(verify_item, statement_list))

        return[item for item in results if item is not None]