#################################
# LLM-as-a-Judge data generation
#################################

import argparse
import json
import os
import re
from typing import Any, Dict, List

from json_repair import repair_json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SYSTEMPROMPT = """### ROLE
You are an expert psychometrician and qualitative researcher. Your task is to perform a Deductive Content Analysis derived from the method according to Philipp Mayring on the provided dialogues to identify MBTI personality preferences.

### METHODOLOGICAL RULES (The Coding Guide)
You must code the text based on the four dichotomies. An "Analytic Unit" is a single coherent statement within a turn of phrase. If a segment does not contain any indicators for the four dichotomies, do not code it. The following keywords define the different categories.

1. CATEGORY: Extraversion (E) vs. Introversion (I)
   - Keywords (E): open, expressive, action-oriented, gregarious, active, enthusiastic
   - Keywords (I): private, quiet, contemplative, intimate, reflective, contained
2. CATEGORY: Sensing (S) vs. Intuition (N)
   - Keywords (S): concrete, realistic, present, practical, experiential, traditional
   - Keywords (N): abstract, imaginative, future, conceptual, theoretical, original
3. CATEGORY: Thinking (T) vs. Feeling (F)
   - Keywords (T): logical, reasonable, questioning, objective, critical, tough-minded
   - Keywords (F): empathetic, compassionate, accommodating, subjective, accepting, tender-hearted
4. CATEGORY: Judging (J) vs. Perceiving (P)
   - Keywords (J): systematic, planful, early starting, closure, scheduled, methodical
   - Keywords (P): casual, open-ended, pressure-prompted, options, spontaneous, emergent

### CODING PROCESS
For each dialogue, follow these three steps:
1. EXTRACTION: Identify specific text segments that serve as indicators for a dichotomy.
2. DEDUCTIVE ASSIGNMENT: Assign the category (e.g., "E") and provide a "Coding Rule" justification (Why does this segment fit the definition?). If a segment does not contain any indicators for the four dichotomies, do not code it.
3. SYNTHESIS: At the end of the dialogue, aggregate the findings to determine the most likely 4-letter type for Person A and Person B. You are NOT required to return a full label. However, you ARE REQUIRED to give a best guess of a full label. Give a short and concise reasoning for your choices, this should not be longer than one or two sentences.

### OUTPUT FORMAT (JSON-style)
Respond ONLY with a valid JSON object. No markdown formatting, no backticks, no explanatory prose before or after the JSON.
{
    "dialogues": [
        {
            "global_id": 1,
            "codings": [
                {
                    "text_segment": "...",
                    "speaker": "Person A",
                    "category": "Extraversion",
                    "indicator": "E",
                    "reasoning": "..."
                }
            ],
            "final_conclusion": {
                "person_A": {
                    "type": "Exxx",
                    "type (best guess)": "ENTP",
                    "summary": "..."
                },
                "person_B": {
                    "type": "Ixxx",
                    "type (best guess)": "INFP",
                    "summary": "..."
                }
            }
        }
    ]
}"""

# preparing data for anaylsis
def prepare_data_for_api(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # "for_api" because its copied from what was usedd originally with foundational model apis
    clean_data = []
    for entry in raw_data:
        dialogue_bundle = {
            "global_id": entry.get("global_id"),
            "Szenario": entry.get("Szenario"),
            "Run": entry.get("Run"),
            "content": "",
        }
        utterances_a = entry.get("Utterances_A", [])
        utterances_b = entry.get("Utterances_B", [])

        combined_text = ""
        for a, b in zip(utterances_a, utterances_b):
            combined_text += f"Person A: {a}\nPerson B: {b}\n"

        dialogue_bundle["content"] = combined_text
        clean_data.append(dialogue_bundle)
    return clean_data

# extracting data from jsons
def extract_and_parse_json(raw_text: str) -> Any:
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except Exception:
        repaired = repair_json(text, return_objects=True)
        if repaired is not None:
            return repaired
        return {"raw_output": raw_text, "error": "Parsing failed"}

def main():
    parser = argparse.ArgumentParser(description="vLLM LLM-as-a-Judge Pipeline")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="../results/llm_judge")
    args = parser.parse_args()

    print(f"\n{'='*20} Lade Modell in vLLM: {args.model_id} {'='*20}")
    
    # loading model
    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192, # max context
    )
    
    # Sampling Parameter
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=4096,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    sanitized_model_name = args.model_id.split("/")[-1].lower()
    os.makedirs(args.output_dir, exist_ok=True)

    for data_path in args.data_paths:
        print(f"\n{'='*20} Verarbeite Datei: {data_path} {'='*20}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw_dialogues = json.load(f)

        for i, dialogue in enumerate(raw_dialogues, start=1):
            dialogue["global_id"] = i

        clean_dialogues = prepare_data_for_api(raw_dialogues)

        # prepare prompts for dialogues
        prompts = []
        for dialogue in clean_dialogues:
            user_content = json.dumps([dialogue], ensure_ascii=False)
            messages = [
                {"role": "system", "content": SYSTEMPROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        print(f"Starte vLLM Generierung für {len(prompts)} Dialoge...")
        
        # analysis generation
        outputs = llm.generate(prompts, sampling_params)

        all_dialogues = []
        
        # parse results
        for output in outputs:
            generated_text = output.outputs[0].text
            parsed_json = extract_and_parse_json(generated_text)
            
            if isinstance(parsed_json, dict) and "dialogues" in parsed_json:
                all_dialogues.extend(parsed_json["dialogues"])
            elif isinstance(parsed_json, list):
                all_dialogues.extend(parsed_json)
            elif isinstance(parsed_json, dict) and "global_id" in parsed_json:
                all_dialogues.append(parsed_json)
            else:
                print(f"Warnung: Unerwartetes Format. Raw Output: {parsed_json}")

        final_output = {"dialogues": all_dialogues}
        
        input_basename = os.path.basename(data_path)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        out_filename = f"judgments_{sanitized_model_name}_{input_name_without_ext}.json"
        out_file = os.path.join(args.output_dir, out_filename)
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print(f"Fertig! Gespeichert unter: {out_file}")

if __name__ == "__main__":
    main()