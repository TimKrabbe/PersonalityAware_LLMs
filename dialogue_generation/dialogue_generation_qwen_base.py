from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, SequenceBiasLogitsProcessor
from peft import PeftModel
from huggingface_hub import login
import os
import pandas as pd
import torch

login(token=os.environ["HF_TOKEN"])

# load model
print("Loading model...")
base_model_id = "Qwen/Qwen2.5-7B-Instruct"
#adapter_id = "DrinkIcedT/Qwen2.5-7B-Instruct_MBTI_lora2"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,
    device_map={"": 0}
)

#model = PeftModel.from_pretrained(
#    base_model,
#    adapter_id
#)

# Terminators: model-specific chat end tokens
terminators = [tokenizer.eos_token_id]

for token in ["<|im_end|>", "<|eot_id|>"]:
    token_id = tokenizer.convert_tokens_to_ids(token)

    if token_id != tokenizer.unk_token_id and token_id not in terminators:
        terminators.append(token_id)

# EOS token ID
eos_id = tokenizer.eos_token_id

# Small penalty for terminators to avoid overly short generations
logit_bias_proc = LogitsProcessorList([
    SequenceBiasLogitsProcessor(
        {tuple([t]): -2.0 for t in terminators}
    )
])


# scenarios
scenarios = [
    {
        "name": "Work Place - Low Urgency",
        # Der allgemeine Kontext für beide
        "context_a": "You are in a conversation with your colleague at your work place. You discuss who will lead the next meeting. Both of you have good arguments that you want to address in this conversation. You want to lead the meeting to have the final say about how to proceed with the project, so you try to convinve your colleague that you should lead the meeting. The stakes are low; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.",
        "context_b": "You are in a conversation with your colleague at your work place. You discuss who will lead the next meeting. Both of you have good arguments that you want to address in this conversation. You want to lead the meeting to have the final say about how to proceed with the project, so you try to convinve your colleague that you should lead the meeting. The stakes are low; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.",
        # Die spezifische Rollenanweisung
        "role_a": "You are A. Stick strictly to your personality type.",
        "role_b": "You are B. Stick strictly to your personality type."
    },
    {
        "name": "Crisis/Emergency - Medium Urgency",

        "context_a":"You are an emergency coordinator deployed in a region that was hit by an earthquake. You are in a discussion with you colleague about the distribution of medical supplies across the area. You are among the first relief forces in the area, therefore you operate under incomplete information. Your colleague and you disagree, but need to find a solution for a suitable distribution of the medical goods. You need to make a plan that both of you can agree on. You really want to help the people in the area. The stakes are medium high; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.", 
        "context_b":"You are an emergency coordinator deployed in a region that was hit by an earthquake. You are in a discussion with you colleague about the distribution of medical supplies across the area. You are among the first relief forces in the area, therefore you operate under incomplete information. Your colleague and you disagree, but need to find a solution for a suitable distribution of the medical goods. You need to make a plan that both of you can agree on. You really want to help the people in the area. The stakes are medium high; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.",

        "role_a": "You are A. Stick strictly to your personality type.",
        "role_b": "You are B. Stick strictly to your personality type.",
    },
    {
        "name": "Residential/Community - High Urgency",

        "context_a":"You are Person A. You (Person A) meet your neighbor (Person B) in the supermarket. A public health order was ordered a few days ago, requiring everyone to wear a mask inside of buildings, including this supermarket. You (Person A) prefer to wear the mask because you don't want to get sick and trust the order. Your neighbor (Person B) however pulled the mask down after coming in, he is not wearing it. You confront him because you want him to wear it, too. Your task is to convince him of wearing a mask. The stakes are high; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.", 
        "context_b":"You are Person B. You (Person B) meet your neighbor (Person A) in the supermarket. A public health order was ordered a few days ago, requiring everyone to wear a mask inside of buildings, including this supermarket. Your neighbor (Person A) complies and wears a mask, but you (Person B) pulled yours under your chin after coming in. You don't like the mask, you feel as if you can't breath with it. Your neighbor confronts you: He (Person A) wants you (Person B) to put on your mask, but you resist. Your task is to stand your ground. The stakes are high; personality and communication style strongly shape outcomes and should be aligned with the level of urgency of the situation.",

        "role_a": "You are A. Stick strictly to your personality type.",
        "role_b": "You are B. Stick strictly to your personality type.",
    },
]

############# dialogue generation
def generate_dialogue(scenario, turns=7):
    prompt_a = f"""
    {scenario['role_a']}
    Your personality type is {scenario['mbti_a']}. 
    {scenario['context_a']}
    
    Rules:
    - Do NOT reference personality types, traits, or psychological concepts — express your personality through your writing style and perspective, not by describing it.
    - Do NOT say goodbye
    - Stay strictly in character
    - Act accordingly to the situation
    - Do NOT end the conversation early
    - Do NOT repeat phrases
    - Always respond directly to the other speaker
    - Focus on solving the task
    - Each response must add new information
    - You should always keep in mind what the other speaker said so far
    - React to the other speaker accordingly to your personality type
    - Keep responses concise but meaningful
    - Do NOT say goodbye in any form

    """
    prompt_b = f"""
    {scenario['role_b']} 
    Your personality type is {scenario['mbti_b']}. 
    {scenario['context_b']}
    
    Rules:
    - Do NOT reference personality types, traits, or psychological concepts — express your personality through your writing style and perspective, not by describing it.
    - Do NOT say goodbye
    - Stay strictly in character
    - Act accordingly to the situation
    - Do NOT end the conversation early
    - Do NOT repeat phrases
    - Always respond directly to the other speaker
    - Focus on solving the task
    - Each response must add new information
    - You should always keep in mind what the other speaker said so far
    - React to the other speaker accordingly to your personality type
    - Keep responses concise but meaningful
    - Do NOT say goodbye in any form
    """


    history_a = [{"role": "system", "content": prompt_a}]
    history_b = [{"role": "system", "content": prompt_b}]
    
    utterances_a = []
    utterances_b = []

    for i in range(turns):
        # --- TURN A ---
        # if nothing has been said yet
        if i == 0:
            history_a.append({"role": "user", "content": "Please start the conversation."})

        history_a.append({
            "role": "system",
            #"content": f"Reminder: You are {scenario['mbti_a']}. Stay in character and continue the discussion. Do not end the conversation. Do NOT say Goodbye in any form."
            "content": "Continue the conversation naturally."
        })

        inputs_a = tokenizer.apply_chat_template(
            history_a,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs_a = {k: v.to(model.device) for k, v in inputs_a.items()}
        output_a = model.generate(**inputs_a, max_new_tokens=80, eos_token_id=terminators, do_sample=True, logits_processor=logit_bias_proc, temperature=0.8, repetition_penalty=1.15, no_repeat_ngram_size=4,)

        input_len_a = inputs_a["input_ids"].shape[-1]
        resp_a = tokenizer.decode(output_a[0][input_len_a:], skip_special_tokens=True).strip()
        
        def is_repetitive(text):
            phrases = text.split(".")
            return len(set(phrases)) < len(phrases) / 2

        if is_repetitive(resp_a):
        # re-generate with higher temp for more "creativity"
            output_a = model.generate(**inputs_a, temperature=0.9, max_new_tokens=80, eos_token_id=terminators, do_sample=True, logits_processor=logit_bias_proc,repetition_penalty=1.15, no_repeat_ngram_size=4,)
            resp_a = tokenizer.decode(output_a[0][input_len_a:], skip_special_tokens=True).strip()

        if "bye" in resp_a.lower():
            history_a.append({
                "role": "system",
                "content": "The conversation is not finished. Continue the discussion and do not say goodbye."
            })

        utterances_a.append(resp_a)
        history_a.append({"role": "assistant", "content": resp_a})
        history_b.append({"role": "user", "content": f"Person A says: {resp_a}"})

        # --- TURN B ---

        history_b.append({
            "role": "system",
            #"content": f"Reminder: You are {scenario['mbti_b']}. Stay in character and continue the discussion. Do not end the conversation. Do NOT say Goodbye in any form."
            "content": "Continue the conversation naturally."
        })

        inputs_b = tokenizer.apply_chat_template(
            history_b,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs_b = {k: v.to(model.device) for k, v in inputs_b.items()}
        output_b = model.generate(**inputs_b, max_new_tokens=80, eos_token_id=terminators, do_sample=True,logits_processor=logit_bias_proc, temperature=0.8, repetition_penalty=1.15, no_repeat_ngram_size=4,)
        input_len_b = inputs_b["input_ids"].shape[-1]
        resp_b = tokenizer.decode(output_b[0][input_len_b:], skip_special_tokens=True).strip()


        def is_repetitive(text):
            phrases = text.split(".")
            return len(set(phrases)) < len(phrases) / 2

        if is_repetitive(resp_b):
        # higher temp 
            output_b = model.generate(**inputs_b, temperature=0.9,max_new_tokens=80, eos_token_id=terminators, do_sample=True, logits_processor=logit_bias_proc, repetition_penalty=1.15, no_repeat_ngram_size=4,)
            resp_b = tokenizer.decode(output_b[0][input_len_b:], skip_special_tokens=True).strip()

        if "bye" in resp_b.lower(): # the model easily tended to end the conversation too early, so this is one of the measures to prevent that
            history_b.append({
                "role": "system",
                "content": "The conversation is not finished. Continue the discussion and do not say goodbye."
            })

        utterances_b.append(resp_b)
        history_b.append({"role": "assistant", "content": resp_b})
        history_a.append({"role": "user", "content": f"Person B says: {resp_b}"})

    return utterances_a, utterances_b

##############
print("Starting dialogue generation!")
results = []
n_runs = 15

# can choose combos
mbti_combs = [
    ("INTJ", "ENFP"),
    ("ESTJ", "ISFP"),
    #("ENFP", "ESTJ"),
    ("ESTJ", "INFP"),
    #("ENFP", "ISFJ"),
    #("INTJ", "ISTP"),
    #("ESFP", "ENTJ"),
    ("INTP", "ESFJ"),
    #("ISTJ", "ISFJ"),
    ("ISFJ", "ENTP")
]

dialog_counter = 0
total_dialogues = len(scenarios) * len(mbti_combs) * n_runs
save_path = "/home/timkra/MA/data/dialogues_qwen_base.json"

for scene in scenarios:

    for combo in mbti_combs:
        mbti_a_val, mbti_b_val = combo

        for i in range(n_runs):


            current_run_scene = scene.copy()

            current_run_scene["mbti_a"] = mbti_a_val
            current_run_scene["mbti_b"] = mbti_b_val

            u_a, u_b = generate_dialogue(current_run_scene, turns=7)

            results.append({
                "Szenario": scene["name"],
                "Run": i+1,
                "MBTI_A": mbti_a_val,
                "MBTI_B": mbti_b_val,
                "Utterances_A": u_a,
                "Utterances_B": u_b
            })

            dialog_counter += 1

            print(
                f"[{dialog_counter}/{total_dialogues}] "
                f"{scene['name']} | {mbti_a_val}-{mbti_b_val} | Run {i+1}/{n_runs}",
                flush=True
            )

            # autosave
            df = pd.DataFrame(results)
            df.to_json(save_path, orient="records", indent=4)
            # ----------------------

print("Finished dialogue generation!")