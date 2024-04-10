from pydantic import BaseModel
from typing import Any, Dict
from peft import AutoPeftModelForCausalLM
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline, GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
import re
from llama_cpp import Llama
from hansolrag.augment import Augment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator(BaseModel):
    mode: str
    model: Any
    tokenizer: Any
    generation_config: Dict[str, Any]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):

        options = config.get('generation-model', [])
        if not options:
            raise ValueError("No generation-model options found in the config")
        
        chosen_option = options[0]
        mode = chosen_option['name']

        if chosen_option is None:
            raise ValueError("Please enter the model-name, only three mode names are supported: 'gpu-solar;, 'cpu-gpt2' and 'cpu-gemini'")

        model = cls.get_generative_model(cls, mode, chosen_option['model_id'], chosen_option.get('quantized', False))
        tokenizer = cls.get_tokenizer(cls, mode, chosen_option['model_id'], chosen_option.get('tokenizer_config', {}))
        generation_config = config.get('generation-config', {})

        return cls(
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config
        )
        

    def get_generative_model(self, mode: str, model_id: str, quantized: str):
        if mode == "gpu-solar":
            if quantized:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.half
                )
            else:
                bnb_config = None

            logger.info("Loading 4 bit-quantized foundation model %s ...", model_id)

            model = AutoPeftModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config = bnb_config,
                device_map="auto" #"cuda" if low_cpu_mem_usage=False
            )

            logger.info("Loading PEFT adapters to %s ...", model_id)
            merged_model = model.merge_and_unload()
            logger.info("Succesfully loaded model %s.", model_id)

            return merged_model
        
        elif mode == "cpu-gpt2":
            logger.info("Loading model %s ...", model_id)
            model =  GPT2LMHeadModel.from_pretrained(model_id)
            logger.info("Succesfully loaded model %s.", model_id)
            return model
    
        elif mode == "cpu-gemini":
            logger.info("Loading model %s ...", model_id)
            model = Llama(model_path=model_id, n_ctx=1000)
            logger.info("Succesfully loaded model %s.", model_id)
            return model
        
        else:
            raise ValueError("Currently, only three mode names are supported: 'gpu-solar;, 'cpu-gpt2' and 'cpu-gemini'")


    def get_tokenizer(self, mode, model_id, tokenizer_config):
        if mode=="gpu-solar":
            if tokenizer_config.get("eos_token", None) is not None:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, eos_token=tokenizer_config.get("eos_token"))
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            return tokenizer
        
        elif mode=="cpu-gpt2":
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, **tokenizer_config)
            return tokenizer
        
        elif mode=="cpu-gemini":
            return None

        else:
            raise ValueError("Currently, only three mode names are supported: 'gpu-solar;, 'cpu-gpt2' and 'cpu-gemini'")

    
    def generate_text(self, text):
        if self.mode=="gpu-solar" or self.mode=="cpu-gpt2":
            pipeline_ = pipeline("text-generation", self.model, tokenizer=self.tokenizer)
            logger.info("Generation config being used: %s.", str(self.generation_config))
            output = pipeline_(text,
                    eos_token_id = self.tokenizer.eos_token_id,
                    **self.generation_config
                    )
            return output[0].get('generated_text').lstrip().rstrip()
        
        elif self.mode=="cpu-gemini":
            max_tokens = self.generation_config.get('max_new_tokens', 256)
            logger.info("Generation config being used: %s.", str({'max_tokens': max_tokens}))
            output = self.model(text, max_tokens=max_tokens)
            return output.get('choices')[0].get('text').lstrip('*').rstrip('*')

        else:
            raise ValueError("Currently, only three mode names are supported: 'gpu-solar;, 'cpu-gpt2' and 'cpu-gemini'")


    def get_output(self, retriever, query_list):
        responses = []

        logger.info("Generating responses ...")

        for idx, question in tqdm(enumerate(query_list)):
            questions = re.split('[?!.]', question)
            seperate_output = []
            prev_q = ""
            for q in questions:
                if len(q) <= 2:
                    continue
                
                prompt_sample = Augment.get_prompt(self.mode, retriever, q + "?", prev_q=prev_q).prompt
                logger.debug("Prompt_sample: %s", prompt_sample)

                output = self.generate_text(prompt_sample)
                seperate_output.append(output)
                prev_q = q

            answer = " ".join(seperate_output)

            logger.info("Question: %s", question)
            logger.info("Response: %s", answer)

            responses.append(answer)

        logger.info("Succesfully generated responses.")

        return responses
