import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: PrefixSpan is a symbolic data mining algorithm. 
# In a real pipeline, you would run this once on your dataset 
# to identify the top K patterns before training the CNN.

class SPM_CN(nn.Module):
    def __init__(self, num_patterns, max_seq_len, num_classes=1):
        super(SPM_CN, self).__init__()
        self.K = num_patterns  # Top K patterns from PrefixSpan [cite: 150]
        self.L = max_seq_len   # Document length 
        
        # --- PHASE 3: Convolutional Neural Network (CNN) ---
        # We apply 1D Convolutions along the time axis 
        # Filter size is 3 x K as specified in the derivation 
        self.conv1d = nn.Conv1d(
            in_channels=self.K, 
            out_channels=32, 
            kernel_size=3, 
            padding=1
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, pattern_image):
        """
        Args:
            pattern_image: Binary matrix Mi of shape (batch, L, K) 
        """
        # Phase 3: CNN Processing
        # Permute to (batch, K, L) to treat patterns as 'channels' for 1D Conv
        x = pattern_image.permute(0, 2, 1) 
        
        # Apply 1D Convolution 
        x = F.relu(self.conv1d(x))
        
        # Global pooling and classification
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        
        return torch.sigmoid(logits)

# --- PHASE 2: Pattern-to-Image Transformation ---
def create_pattern_image(text_tokens, top_patterns, max_len):
    """
    Transforms a document into a binary matrix Mi[cite: 151, 152].
    Mi[t, k] = 1 if pattern pk ends at position t[cite: 153].
    """
    L = max_len
    K = len(top_patterns)
    # Initialize binary matrix with zeros 
    matrix = torch.zeros((L, K))
    
    for k, pattern in enumerate(top_patterns):
        p_len = len(pattern)
        # Sliding window to find where the pattern ends
        for t in range(p_len, len(text_tokens) + 1):
            window = text_tokens[t-p_len : t]
            if window == pattern:
                if t-1 < L:
                    matrix[t-1, k] = 1  # Set bit if pattern ends at t [cite: 153]
                    
    return matrix