# Specification

This document defines a handwritten digit recognition program for the University of Helsinki Algorithms and AI Project course. I am completing this course in the Bachelor’s in Computer Science (TKT) program. This project’s documentation will be written in English.

I am using Python, also proficient in Typescript and Javascript.

I will implement a neural network structure, consisting of sigmoid neurons, and a stochastic gradient descent learning algorithm for solving the classification of grayscale images of handwritten digits from the MNIST dataset into one of 10 categories.

The inputs the program receives are 28x28 px images as vectors which are fed into the neural network and integers 0-9 which will be one-hot-encoded to vectors for training. The expected time complexity for most stages of the program is O(l _ n²) and the space complexity is O(l _ n) when l = layers in the neural network and n = neurons per layer.

Some of the resources I intend to use:

- https://www.youtube.com/watch?v=sIX_9n-1UbM
- https://blog.rlamsal.com.np/forward-pass-backpropagation-example/
- https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- http://neuralnetworksanddeeplearning.com/chap1.html
- https://www.deeplearningbook.org/
- https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista
- https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/
- https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/
- https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
- https://materiaalit.github.io/intro-to-ai/
