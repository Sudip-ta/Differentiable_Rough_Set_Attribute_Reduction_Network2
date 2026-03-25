import torch
import torch.nn as nn

class RoughSetFeatureSelector(nn.Module):
    def __init__(self, num_features, sigma=1.0, alpha=10.0):
        super(RoughSetFeatureSelector, self).__init__()
        # Step 4.1.2 - Step 1: Learnable weights w
        self.theta = nn.Parameter(torch.ones(num_features)) 
        self.sigma = sigma
        self.alpha = alpha # This is for the "Smooth Max" approximation

    def get_weights(self):
        return torch.sigmoid(self.theta)

    def forward(self, x, labels=None):
        weights = self.get_weights()
        x_weighted = x * weights
        
        # If we are just running the model, return the weighted features
        if labels is None:
            return x_weighted, weights

        # Step 4.1.2 - Step 2: Differentiable Indiscernibility Relation (R)
        # Calculate Weighted Euclidean Distance
        dist = torch.cdist(x_weighted, x_weighted, p=2)
        R = torch.exp(-(dist**2) / (2 * self.sigma**2))

        # Step 4.1.2 - Step 3: Soft Lower Approximation (mu)
        # This part calculates how 'certain' we are about each sample
        num_samples = x.shape
        mu_list = []
        
        for i in range(num_samples):
            # Find indices of samples NOT in the same class (Step 3: j not in Cc)
            different_class_mask = (labels != labels[i])
            
            if different_class_mask.any():
                # Smooth Maximum approximation (SoftMin logic)
                enemy_sims = R[i, different_class_mask]
                # Applying the SoftMax formula from your screenshot:
                # max(Rij) approx sum(Rij * exp(alpha*Rij)) / sum(exp(alpha*Rij))
                numerator = torch.sum(enemy_sims * torch.exp(self.alpha * enemy_sims))
                denominator = torch.sum(torch.exp(self.alpha * enemy_sims))
                max_sim = numerator / (denominator + 1e-8)
                
                mu_list.append(1.0 - max_sim)
            else:
                mu_list.append(torch.tensor(1.0, device=x.device))

        # Step 4.1.2 - Step 4: Dependency Degree (gamma)
        gamma = torch.stack(mu_list).mean()
        
        # L_rs = 1 - gamma
        rough_set_loss = 1.0 - gamma
        
        return x_weighted, weights, rough_set_loss
    
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
    
class DRSAR_FL_RNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes=1):
        super(DRSAR_FL_RNN, self).__init__()
        
        # 1. The Filter (Your first code logic)
        self.selector = RoughSetFeatureSelector(num_features)
        
        # 2. The Brain (Your second code logic)
        self.rnn = FuzzyRNN(num_features, hidden_dim)

    def forward(self, x_sequence):
        # x_sequence shape: (batch, time_steps, features)
        
        # --- PHASE 1: Feature Selection ---
        # Get the current Sigmoid weights (theta)
        weights = self.selector.get_weights()
        
        # Apply weights to EVERY day in the sequence
        # (batch, time, features) * (features)
        x_filtered = x_sequence * weights
        
        # --- PHASE 2: Sequence Processing ---
        # Feed the 'Cleaned' data into the Fuzzy RNN
        logits = self.rnn(x_filtered)
        
        # Return both so we can calculate the Rough Set Loss later!
        return logits, weights