from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
import torch
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from huggingface_hub import login
import os
import mlflow
from dataclasses import dataclass

# HF Token login
login(token=os.environ["HF_TOKEN"])

# import data
print("Loading dataset")
df = load_dataset("DrinkIcedT/mbti_dialogue_pub_filtered")

# mlflow
print("Setting up mlflow")
mlf_exp_name = "Finetuning_Ministral-8B_lora1"
mlflow_path = "/home/timkra/MA/mlflow/mlflow_ministral_lora1.db"

output_dir = os.path.join(
    os.environ["SCRATCH_DIR"],
    "Ministral-8B-Instruct-MBTI_lora1"
)

# paras
# LoRA
@dataclass
class LoRA_Params:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

# model
@dataclass
class Model_Params:
    lr: float = 2e-5
    max_grad_norm: float = 1.0
    n_epochs: int = 6
    warmup_steps: float = 0.03
    batch_size: int = 8
    gradient_acc_steps: int = 2
    num_gpus: int = 4

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_acc_steps * self.num_gpus

lora_params = LoRA_Params()
model_params = Model_Params()

# model
print("Setting up model!")
model_checkpoint = "mistralai/Ministral-8B-Instruct-2410"
model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_tokenizer.padding_side = "right"

if model_tokenizer.pad_token is None:
    model_tokenizer.pad_token = model_tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)


# LoRA
lora_config = LoraConfig(
    r=lora_params.r,
    lora_alpha=lora_params.lora_alpha,
    lora_dropout=lora_params.lora_dropout,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)


# training args
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=model_params.batch_size,
    num_train_epochs=model_params.n_epochs,
    gradient_accumulation_steps=model_params.gradient_acc_steps,
    learning_rate=model_params.lr,
    warmup_steps=model_params.warmup_steps,
    max_grad_norm=model_params.max_grad_norm,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps= 200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    report_to="tensorboard",
    max_length=512,
)

# trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=df["train"],
    eval_dataset=df["validation"],
    peft_config=lora_config,
    processing_class=model_tokenizer,  # übernimmt Tokenisierung
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

if trainer.is_world_process_zero():
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_path}")
    mlflow.set_experiment(mlf_exp_name)

    print("Starting mlflow run!")
    mlflow.start_run()
    mlflow.log_param("LoRA: Rank", lora_params.r)
    mlflow.log_param("LoRA: Alpha", lora_params.lora_alpha)
    mlflow.log_param("LoRA: Dropout", lora_params.lora_dropout)

    mlflow.log_param("Model: Number of Epochs", model_params.n_epochs)
    mlflow.log_param("Model: Learning Rate", model_params.lr)
    mlflow.log_param("Model: Warm-up Steps", model_params.warmup_steps)
    mlflow.log_param("Model: Max Gradient Norm", model_params.max_grad_norm)
    mlflow.log_param("Model: Batch Size per device", model_params.batch_size)
    mlflow.log_param("Model: Gradient Accumulation Steps", model_params.gradient_acc_steps)
    mlflow.log_param("Model: Effective Batch Size", model_params.effective_batch_size)

try:
    print("Starting training!")
    trainer.train()

    if trainer.is_world_process_zero():
        print("Saving model in home dir")
        trainer.save_model("/home/timkra/MA/output/LoRAadapter/Ministral_lora1")
        print("Pushing model to hub!")
        trainer.push_to_hub("DrinkIcedT/Ministral-8B_MBTI_lora1")

finally:
    if trainer.is_world_process_zero():
        print("Ending mlflow run!")
        mlflow.end_run()
