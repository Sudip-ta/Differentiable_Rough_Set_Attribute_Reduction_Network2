import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERT_FIS(nn.Module):
    def __init__(self, num_rules=3):
        super(BERT_FIS, self).__init__()
        # Step 1: BERT Encoding (The Librarian)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_dim = 768 # Standard BERT size
        self.K = num_rules
        
        # Step 2: Type-2 Fuzzy Parameters (Upper and Lower)
        self.c = nn.Parameter(torch.randn(self.hidden_dim, self.K))
        self.sigma_upper = nn.Parameter(torch.ones(self.hidden_dim, self.K) * 1.2)
        self.sigma_lower = nn.Parameter(torch.ones(self.hidden_dim, self.K) * 0.8)
        self.q = nn.Parameter(torch.randn(self.hidden_dim, self.K))

    def forward(self, input_ids, attention_mask):
        # Step 1: Get BERT Features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :] # Use [CLS] token
        
        # Step 2: Interval Type-2 Fuzzification
        x_ext = x.unsqueeze(-1).repeat(1, 1, self.K)
        # Upper membership (mu_u) and Lower membership (mu_l)
        mu_u = torch.exp(-((x_ext - self.c)**2) / (2 * self.sigma_upper**2))
        mu_l = torch.exp(-((x_ext - self.c)**2) / (2 * self.sigma_lower**2))
        
        # Step 3 & 4: Type-Reduction and Defuzzification
        # We average the Upper and Lower strengths
        mu_avg = (mu_u + mu_l) / 2
        f_strength = torch.sum(mu_avg * self.q, dim=-1) / (torch.sum(mu_avg, dim=-1) + 1e-8)
        
        return torch.sigmoid(f_strength)