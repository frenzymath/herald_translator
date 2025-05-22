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

import re
import json


class Util:
    @staticmethod
    def chat_template_to_prompt(messages: list[dict], model: str) -> str:
        """
        Chat template for deepseek and internlm
        """
        result = ""
        total_step = len(messages)
        for i, message in enumerate(messages):
            if model == "internlm":
                result += "<|im_start|>" + message["role"] + "\n" + message["content"]
                if i + 1 != total_step:
                    result += "<|im_end|>\n"
                elif message["role"] == "user":
                    result += "<|im_end|>\n<|im_start|>assistant\n"

            elif model == "deepseek":
                if message["role"] == "user":
                    result += "User:" + message["content"] + "\n\n"
                elif message["role"] == "assistant":
                    result += "Assistant" + message["content"] + "<｜end▁of▁sentence｜>"
                elif message["role"] == "system":
                    result += message["content"] + "\n\n"
                if i + 1 == total_step and message["role"] == "user":
                    result += "Assistant:"
            else:
                raise NotImplementedError
        return result

    @staticmethod
    def get_openai_messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def remove_informal_prefix(formal_statement: str) -> str:
        pattern = r"/-- [\s\S]*? -/\n"
        cleaned_text = re.sub(pattern, "", formal_statement, flags=re.DOTALL)
        return cleaned_text

    @staticmethod
    def extract_bold_text(output):
        match = re.search(r"\|\|(.*?)\|\|", output)
        if match:
            return match.group(1)
        return "null"

    @staticmethod
    def jsonltojson(path: str, output_path: str):
        data = []
        with open(path) as fp:
            for l in fp.read().splitlines():
                data.append(json.loads(l))
        with open(output_path, "w") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)


class StatUtil:
    @staticmethod
    def get_verified_stat(output_path):
        with open(output_path) as fp:
            data = json.load(fp)
        vcount = 0
        for d in data:
            if d["verified"]:
                vcount += 1
        print(f"verified: {vcount}\ntotal: {len(data)}\nratio: {vcount / len(data)}")

    @staticmethod
    def get_translated_stat(output_path):
        with open(output_path) as fp:
            data = json.load(fp)
        vcount = 0
        for d in data:
            if d["translated"]:
                vcount += 1
        print(f"translated: {vcount}\ntotal: {len(data)}\nratio: {vcount / len(data)}")
