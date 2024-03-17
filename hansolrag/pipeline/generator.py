from pydantic import BaseModel

from peft import AutoPeftModelForCausalLM
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline
from tqdm import tqdm
import re


class Generator(BaseModel):
    model_id: str = "bibekyess/solar-checkpoint-2000"
    eos_token: str = '###'
    quantized: bool = True

    @property
    def get_generative_model(self):

        if self.quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None

        model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config = bnb_config,
            device_map="auto" #"cuda" if low_cpu_mem_usage=False
        )

        merged_model = model.merge_and_unload()
        return merged_model
    
    @property
    def get_tokenizer(self):
        if eos_token is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, eos_token=self.eos_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
             
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    

    def get_output(self, retriever, query_list):
        text_pipe = pipeline("text-generation", self.model, tokenizer=self.tokenizer)

        responses = []

        for idx, question in tqdm(enumerate(query_list)):
            questions = re.split('[?!.]', question)
            seperate_output = []
            prev_q = ""
            for q in questions:
                if len(q) <= 2:
                    continue

            prompt_sample = Augment.get_prompt(retriever, q + "?", prev_q=prev_q).prompt
            output = text_pipe(prompt_sample,
                                min_new_tokens=20,
                                max_new_tokens=256,
                                top_p=0.98,
                                top_k=50,
                                temperature=0.9,
                                return_full_text=False,
                                eos_token_id = [27332]
                                )
            output = output[0].get('generated_text').lstrip().rstrip()
            seperate_output.append(output)
            prev_q = q

            answer = " ".join(seperate_output)
            responses.append(answer)

        return responses
