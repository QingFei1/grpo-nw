import re
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config



SYSTEM_PROMPT = """Respond in the following format:

<think>
...
</think>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def calculate_similarity(emb_model, prediction, reference):

    embedding_1 = emb_model.encode(prediction, convert_to_tensor=True)
    embedding_2 = emb_model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    
    return float(similarity[0][0])*2


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    example_response = completions[0][0]["content"]
    example_answer = answer[0]
    example_prompt = prompts[0][-1]['content']
    
    score=[calculate_similarity(emb_model,r,a) for r, a in zip(extracted_responses, answer)]
    print(
        f"-" * 50
        + f"\nExample prompt:\n{example_prompt}\n"
        + f"-" * 10
        + f"\nExample response:\n{example_response}\n"
        + f"-" * 10
        + f"\nExample answer:\n{example_answer}\n"
        + f"-" * 10
        + f"\nscore : {score}\n"
        + f"-" * 10
    )
    return score

def answer_format_reward_func(completions, **kwargs) -> list[float]:

    pattern = r"^是否转供电：(是|否)\n一次设备安全措施：\n应断开：[\s\S]*?\n应合上：[\s\S]*$"
    responses = [completion[0]["content"] for completion in completions]

    matches = []

    for response in responses:

        cleaned_response = response.strip()
        match = re.match(pattern, cleaned_response, flags=re.DOTALL)
        
        if match:
            all_parts_present = (
                "是否转供电" in cleaned_response and
                "一次设备安全措施" in cleaned_response and
                "应断开" in cleaned_response and cleaned_response.split("应断开：")[1].strip() and
                "应合上" in cleaned_response and cleaned_response.split("应合上：")[1].strip()
            )
            matches.append(1.0 if all_parts_present else 0.0)
        else:
            matches.append(0.0)
    print("answer_format",matches)
    return matches


def think_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r.strip(), flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]


reward_funcs=[
    think_format_reward_func,
    answer_format_reward_func,
    correctness_reward_func]



def load_data(data_path, split="train") -> Dataset:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    formatted_data = [{
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["instruction"] + 
                                      (f"\n{item['input']}" if item['input'] else "")},
        ],
        "answer": item["output"]
    } for item in data]

    dataset = Dataset.from_list(formatted_data)
    return dataset

def main(training_args, model_args):
    data = load_data("./data/power/power_maintenance_sft_dataset_0313.json")
    train_size = len(data) - 50 
    train_dataset = data.select(range(train_size))
    eval_dataset = data.select(range(train_size, len(data)))


    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in [
        'auto', None
    ] else getattr(torch, model_args.torch_dtype))
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs
     # Initialize tokenizer (adds pad token).
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    emb_model = SentenceTransformer('/workspace/qingfei_new/model/sentencetransformer/mmarco-mMiniLMv2-L12-H384')
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)
