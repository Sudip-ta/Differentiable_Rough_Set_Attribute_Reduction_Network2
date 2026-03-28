import torch
import torch.nn as nn

class DRSARNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DRSARNet, self).__init__()
        
        # This is the "Trainable Weights theta"
        self.theta = nn.Parameter(torch.ones(num_features)) 
        # Hidden Layer 1: Takes input, outputs 64 "thoughts"
        self.layer1 = nn.Linear(num_features, 64)
        
        # Hidden Layer 2: Takes those 64 thoughts, narrows them to 32
        self.layer2 = nn.Linear(64, 32)
        
        # Output Layer: Narrows 32 thoughts into the final classes (e.g., 2 classes)
        self.output_layer = nn.Linear(32, num_classes)
        
        # Dropout: Randomly ignores 50% of neurons during training
        # WHY: To make the model "tougher" so it doesn't just memorize one student.
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Step 1: Feature Weighting
        w = torch.sigmoid(self.theta)
        x_weighted = x * w
        
        # Step 2: Pass through Hidden Layer 1 + ReLU (Activation)
        # WHY: ReLU turns negative numbers to 0, helping the network learn patterns.
        out = torch.relu(self.layer1(x_weighted))
        out = self.dropout(out)
        
        # Step 3: Pass through Hidden Layer 2 + ReLU
        out = torch.relu(self.layer2(out))
        out = self.dropout(out)
        
        # Step 4: Final output (Logits)
        logits = self.output_layer(out)
        
        return logits, w
    
def calculate_rough_set_loss(x, weights, labels, sigma=1.0):
        # This function will calculate how 'certain' the model is 
        # about its feature selection.
        # 1. Multiply features by weights to get 'Weighted Features'
        x_weighted = x * weights
    
        # 2. Calculate the distance between every pair of students
        #We use 'p=2' for Euclidean distance.
        dist_sq = torch.cdist(x_weighted, x_weighted, p=2)**2
        # 3. Apply the Gaussian Kernel formula
        # Rij = exp(-dist^2 / 2*sigma^2)
        R = torch.exp(-dist_sq / (2 * sigma**2))
        num_samples = x.size(0)
        certainty_scores = []
    
        for i in range(num_samples):
            # Find who is NOT in the same class as student 'i'
            mask = (labels != labels[i])
        
        if mask.any():
            # Find the max similarity to anyone in a DIFFERENT class
            max_other_sim = torch.max(R[i, mask])
            
            # Certainty = 1 - (how much I look like an outsider)
            certainty_scores.append(1 - max_other_sim)
        else:
            # If everyone is in the same class, certainty is 100%
            certainty_scores.append(torch.tensor(1.0))
        # Average certainty across the whole batch
        gamma = torch.stack(certainty_scores).mean()
    
        # Return the Loss: 1 - Dependency Degree
        return 1 - gamma
def train_model(data, labels, epochs=100):
        for epoch in range(epochs):
            # A. Clear the memory
            # WHY: We don't want the errors from the last round to mess up this round.
            optimizer.zero_grad() 
        
            # B. Forward Pass (The Model's Guess)
            # WHY: The model looks at the data 'x' and gives its predictions and weights 'w'.
            predictions, w = model(data)
        
            # C. Calculate the Total Loss
            # 1. Classification Error (Cross Entropy)
            l_ce = torch.nn.functional.cross_entropy(predictions, labels)
        
            # 2. Rough Set Error (Dependency Loss)
            l_rs = calculate_rough_set_loss(data, w, labels)
        
            # 3. L1 Regularization (The 'Tax' for using features)
            l_l1 = torch.sum(torch.abs(w))
        
            # Final Total Loss Formula
            total_loss = l_ce + (lambda1 * l_rs) + (lambda2 * l_l1)
        
            # D. Backpropagation (The 'Learning')
            # WHY: Math moves backward to find which weights to nudge.
            total_loss.backward() 
        
            # E. Update the Weights
            # WHY: This is where theta actually changes!
            optimizer.step()
        
            # Print progress every 10 rounds
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f}")
    
# 1. Create the model
# Let's assume we have 4 features (M=4) and 2 classes (C=2)
model = DRSARNet(num_features=4, num_classes=2)

# 2. Setup the Optimizer (The 'Updater')
# AdamW is a very smart version of Gradient Descent mentioned in your chart
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

# 3. Define the 'Loss Weights' (lambda)
# WHY: This decides how much we care about the Rough Set part vs. the L1 Tax
lambda1 = 0.1  # RS loss weight
lambda2 = 0.01 # L1 tax rate

# 1. Create 10 fake students with 4 features each
# (e.g., GPA, Coding, Age, Color)
fake_data = torch.randn(10, 4) 

# 2. Create 10 fake labels (0 or 1)
fake_labels = torch.randint(0, 2, (10,)) 

# 3. Run the training!
print("Starting training with fake data...")
train_model(fake_data, fake_labels, epochs=50)


    
