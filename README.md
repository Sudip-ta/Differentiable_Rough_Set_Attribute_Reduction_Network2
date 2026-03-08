# Differentiable_Rough_Set_Attribute_Reduction_Network2

1. torch.ones(num_features) (The Starting Point)
This is a standard PyTorch function that creates a list (called a Tensor) of 1s.

If your num_features is 4 (GPA, Coding, Age, Color), it creates: [1.0, 1.0, 1.0, 1.0].

The Logic: We start every feature with a weight of 1.0 because we want the model to start by assuming every piece of information is 100% useful. As it trains, it will learn to lower these numbers for useless features.



nn.Parameter (The "Learnable" Tag)
This is the most critical part for a programmer to understand. In PyTorch, a normal Tensor is just a static list of numbers. By wrapping it in nn.Parameter, you are telling the computer:

"This is a weight that needs to change".

Track Gradients: PyTorch will now keep track of every math operation done to these numbers so it can calculate their "blame" (gradient) when the model makes a mistake.

Optimizer Registration: When you tell the optimizer to look at model.parameters(), it automatically finds this theta and knows it must update it during backpropagation.