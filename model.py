import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class CLAMEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, attn_dim=256, dropout=True, gated=True):
        super(CLAMEncoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity()
        )
        if gated:
            self.attention = Attn_Net_Gated(L=hidden_dim, D=attn_dim, dropout=dropout)
        else:
            self.attention = Attn_Net(L=hidden_dim, D=attn_dim, dropout=dropout)

    def forward(self, h):
        h = self.fc1(h)                  # [N_patches, hidden_dim]
        A, h = self.attention(h)        # [N_patches, 1], [N_patches, hidden_dim]
        A = torch.transpose(A, 1, 0)    # [1, N_patches]
        A = torch.softmax(A, dim=1)
        M = torch.mm(A, h)              # [1, hidden_dim]
        return M


class CLAMReportGenerator(nn.Module):
    def __init__(self, 
                 t5_model_name='t5-small',
                 input_dim=1024,
                 clam_hidden=512,
                 t5_d_model=512):
        super(CLAMReportGenerator, self).__init__()
        
        self.clam_encoder = CLAMEncoder(input_dim=input_dim, hidden_dim=clam_hidden)
        self.projector = nn.Linear(clam_hidden, t5_d_model)  # T5 expects 512-d input
        
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.decoder = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, patch_features, report_text=None):
        """
        patch_features: Tensor [N_patches, 1024]
        report_text: string (ground truth report)
        """
        device = patch_features.device

        # === Encode ===
        slide_embed = self.clam_encoder(patch_features)           # [1, 512]
        projected = self.projector(slide_embed)                   # [1, 512]

        # === Tokenize target text ===
        if report_text is not None:
            labels = self.tokenizer(report_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        else:
            labels = None

        # === Decode ===
        decoder_input = self.tokenizer("generate report:", return_tensors="pt").input_ids.to(device)

        # Custom: manually feed encoder_hidden_states
        output = self.decoder(
            input_ids=decoder_input,
            encoder_outputs=(projected.unsqueeze(1),),  # shape [batch, seq_len, d_model]
            labels=labels
        )

        return output  # contains loss if labels provided, logits otherwise

    def generate(self, patch_features, max_length=256):
        with torch.no_grad():
            slide_embed = self.clam_encoder(patch_features)         # [1, 512]
            projected = self.projector(slide_embed)                 # [1, 512]
            decoder_input = self.tokenizer("generate report:", return_tensors="pt").input_ids.to(patch_features.device)

            output_ids = self.decoder.generate(
                input_ids=decoder_input,
                encoder_outputs=(projected.unsqueeze(1),),
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
