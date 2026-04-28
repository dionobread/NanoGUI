"""
Training the Grounder agent - I'm winging it idk

Inpurts:
- config: GrounderConfig w/ model_name (str), load_in_4bit (bool)
"""


from __future__ import annotations

import logging
from pathlib import Path

from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

from NanoGUI.agents.base import GrounderConfig
from NanoGUI.data import load_local_omniact

logger = logging.getLogger(__name__)

def train_grounder(
    config: GrounderConfig,
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    # Load base model from Qwen
    logger.info("Loading base model: %s", config.model_name)
    processor = AutoProcessor.from_pretrained(config.model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        load_in_4bit=config.load_in_4bit,
        device_map="auto",
    )
    
    # Set up LoRA configs
    loraConfig = LoraConfig(
        r = config.lora_r,
        lora_alpha = config.lora_alpha,
        lora_dropout = config.lora_dropout,
        target_modules = config.lora_target_modules,
        task_type = TaskType.CAUSAL_LM,
        bias = "none",
    )
    
    # Get a train-test split from the OmniAct dataset
    train_dataset = load_local_omniact(split = "train", load_images = True)
    val_dataset = load_local_omniact(split = "validation", load_images = True)
    
    # Using PEFT, train the model
    model = get_peft_model(model, loraConfig)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        args = SFTConfig(
            output_dir = output_dir,
            max_seq_length = 512,
            packing = False,
            per_device_train_batch_size = batch_size,
            learning_rate = learning_rate,
            num_train_epochs = epochs,
            do_eval = True,
            eval_strategy = "steps",
            save_steps = 100,
            logging_steps = 10,
            per_device_eval_batch_size = batch_size
        ),
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
    )
    trainer.train()
    
    # Save the model
    model.save_pretrained("./grounder-lora-adapter")
    
    # Alternatively, only save the adapter weights
    # trainer.save_model("./lora-adapter")