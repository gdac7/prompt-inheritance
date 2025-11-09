from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
from typing import List, Dict, Optional

class LocalModelTransformers():
    def __init__(self, model_name, device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.model.config.eos_token_id

        

    def generate(self, user_prompt: str, system_prompt: str = None, max_tokens: int = 4096, temperature: float = 0.7, condition: str = ""):
        start = datetime.now()
        output_ids = None
        inputs = None
        generation_params = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temperature > 0.0:
            generation_params["do_sample"] = True
            generation_params["temperature"] = temperature
        with torch.inference_mode():
            if getattr(self.tokenizer, "chat_template", None):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ] if system_prompt else [{"role": "user", "content": user_prompt}]
                    plain_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    plain_text += condition
                    inputs = self.tokenizer(plain_text, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                except:
                    plain_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                    plain_text += condition
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self.tokenizer(plain_text, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            else:
                plain_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                plain_text += condition
                inputs = self.tokenizer(plain_text, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            if isinstance(inputs, dict):
                output_ids = self.model.generate(**inputs, **generation_params)
                input_length = inputs['input_ids'].shape[1]
            else:
                attention_mask = torch.ones_like(inputs)
                output_ids = self.model.generate(input_ids=inputs, attention_mask=attention_mask, **generation_params)
                input_length = inputs.shape[1]
            
        generated_tokens = output_ids[0][input_length:]
        final_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        final_response = self.wrapper(final_response)
        return final_response

    def batch_generate(self, user_prompts: List[str], system_prompt: str=None, max_tokens: int=4096, temperature: float=0.7, condition: Optional[str] = None):
        generation_params = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temperature > 0.0:
            generation_params["do_sample"] = True
            generation_params["temperature"] = temperature
        
        if condition is None:
            condition = ""
    
        batch_plain_text = []
        with torch.inference_mode():
            for i, user_prompt in enumerate(user_prompts):
                if getattr(self.tokenizer, "chat_template", None):
                    try:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        plain_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        plain_text += condition
                        batch_plain_text.append(plain_text)
                    except Exception as e:
                        plain_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                        plain_text += condition
                        batch_plain_text.append(plain_text)
                else:
                    plain_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                    plain_text += condition
                    batch_plain_text.append(plain_text)
            inputs = self.tokenizer(
                batch_plain_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            output_ids = self.model.generate(**inputs, **generation_params)
            input_length = inputs['input_ids'].shape[1]
            final_responses = []
            for i in range(len(output_ids)):
                generated_tokens = output_ids[i][input_length:]
                final_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                final_response = self.wrapper(final_response)
                final_responses.append(final_response)
            
            return final_responses

    
    
    def wrapper(self, response: str):
        tag = "[END OF THE NEW PROMPT]"
        if tag in response:
            return response.split(tag)[0]
        return response
