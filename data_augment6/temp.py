from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
import torch

config = RobertaConfig()
config.output_hidden_states = True

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM(config).from_pretrained("roberta-base")

inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # retrieve index of <mask>
# mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
# print(tokenizer.decode(logits[0].argmax(axis=-1)))


with torch.no_grad():
    out = model(**inputs).hidden_states

print(out)