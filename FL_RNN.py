import torch
import torch.nn as nn

class FuzzyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_rules=3):
        super(FuzzyRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.K = num_rules  # Number of Fuzzy Membership Functions (e.g., Cold, Warm, Hot)
        
        # 1. The "Knobs" for Thinking (Wx, Wh, b)
        self.Wx = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        # 2. Fuzzy Parameters (These are LEARNABLE by the robot)
        # c = Center of the cloud, sigma = width, q = the "if-then" result
        self.c = nn.Parameter(torch.randn(hidden_dim, self.K))
        self.sigma = nn.Parameter(torch.ones(hidden_dim, self.K))
        self.q = nn.Parameter(torch.randn(hidden_dim, self.K))
        
        # 3. The Final Prediction Layer (Fire or No Fire?)
        self.classifier = nn.Linear(hidden_dim, 1)

    def fuzzy_gate(self, z):
        # This is Step 4.2.1: The Fuzzy Gate logic
        # z shape: (batch, hidden_dim)
        
        # Step A: Expand z to compare with all K rules
        z_expanded = z.unsqueeze(-1).repeat(1, 1, self.K)
        
        # Step B: Fuzzification (The Gaussian Cloud)
        # mu = exp( -(z - c)^2 / (2 * sigma^2) )
        mu = torch.exp(-((z_expanded - self.c)**2) / (2 * (self.sigma**2) + 1e-8))
        
        # Step C: Fuzzy Inference (The TSK weighted average)
        numerator = torch.sum(mu * self.q, dim=-1)
        denominator = torch.sum(mu, dim=-1) + 1e-8
        
        # We use sigmoid to make sure the gate output is between 0 and 1
        return torch.sigmoid(numerator / denominator)

    def forward(self, x_sequence):
        # x_sequence shape: (batch_size, time_steps, input_dim)
        batch_size, time_steps, _ = x_sequence.size()
        
        # Start with an empty notebook (Memory = 0)
        h = torch.zeros(batch_size, self.hidden_dim).to(x_sequence.device)
        
        # Process each day in the sequence
        for t in range(time_steps):
            x_t = x_sequence[:, t, :]
            
            # Step 1: The Linear Thought (zt = Wx*x + Wh*h + b)
            z_t = torch.matmul(x_t, self.Wx) + torch.matmul(h, self.Wh) + self.b
            
            # Step 2: Ask the Fuzzy Guard (The Gate)
            g_t = self.fuzzy_gate(z_t)
            
            # Step 3: Update the Notebook (The RNN Update formula)
            # Today's news filtered by a tanh (to keep numbers between -1 and 1)
            new_info = torch.tanh(torch.matmul(x_t, self.Wx))
            
            # The Gate decides: How much old memory vs how much new info?
            h = (1 - g_t) * h + g_t * new_info
            
        # After looking at all days, make the final prediction
        return self.classifier(h)