# Multi-Dimensional-Logic-Neural-Net
Neural Net that learns logic operations for multi-dimensional inputs

Visual representation of how the neural network learns the weights to solve logic (specifically XOR) problems (you can modify the output to be any funciton of the inputs and the network will learn to achieve that output)

2-Dimensional inputs:

![alt tag](https://github.com/ConsciousMachines/Multi-Dimensional-Logic-Neural-Net/blob/master/ex2d.png)

3-Dimensional inputs:

![alt tag](https://github.com/ConsciousMachines/Multi-Dimensional-Logic-Neural-Net/blob/master/ex3d.png)

To solve the XOR problem the minimum requirement is 1 hidden layer with 2 hidden units and 1 unit in the output layer. 
For 4-Dimensional inputs you need to add an additional hidden unit for it to work (3 hidden units total)

This s mainly a simple example to help visualize the training process of neural nets, get comfortable with data feature representations, and understand the intuition behind the clever trick that disproved Minsky and Papert's 1-layer Perceptron disproof, by stacking two perceptrons together. 
