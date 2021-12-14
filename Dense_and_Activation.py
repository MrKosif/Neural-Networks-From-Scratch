import numpy as np
from nnfs import spiral_data

input = [[1, -2, 3],
         [-3, 6 ,-8]]

class Layer_Dense:
    def __init__(self, no_of_inputs, no_of_neurons):
        self.weight = 0.10*np.random.randn(no_of_inputs, no_of_neurons)
        self.bias = np.zeros((1, no_of_neurons))

    def forward(self, input):
        output = np.dot(input, self.weight) + self.bias

class Activation_ReLU:
    def forward(self, input):
        output = np.maximum(0, input)
        print(output)

input = [[1, 2, 3, 5, 6, 8, 5],
         [2, 5, 3, 7, 4, 3, 1]]
class Softmax_Activation:
    def forward(self, input):
        # softmaxin yaptığı şey şu: atıyorum 1 e 3 şül bir array var birini al diğerlerinin toplamına böl
        output = input / np.sum(input, axis=1, keepdims=True)
        print(output)

class Catagorical_Crossentrophy:
    # olay su one hot encoding alınacak ve class doğruysa yani birse o classın one hopt 
    # coding ile çarpılır sonra toplanır en son negatıfı alınır
    
    pass
#relu = Activation_ReLU()
#relu.forward(input)
#softi = Softmax_Activation()
#softi.forward(input)
X, y = nnfs.spiral_data(samples=100, classes=3)
print(y)

#layer1 = Layer_Dense(3, 4)
#layer2 = Layer_Dense(4, 5)
#layer2.forward(layer1.forward(input))


         