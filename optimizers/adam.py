import numpy as np


# from .gradientdescent import GradientDescent


# TODO: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for name in layers_list:
            # TODO: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            v = [np.zeros_like(p) for p in layers_list[name].parameters]
            s = [np.zeros_like(p) for p in layers_list[name].parameters]
            self.V[name] = v
            self.S[name] = s


def update(self, grads, name, epoch):
    layer = self.layers[None]
    params = []
    # TODO: Implement Adam update
    for index in range(len(grads)):
        self.V[name][index] = self.beta1 * self.V[name][index] + (1 - self.beta1) * grads[index]
        self.S[name][index] = self.beta2 * self.S[name][index] + (1 - self.beta2) * np.power(grads[index, 2])
        self.V[name][index] /= 1 - self.beta1 ** epoch
        self.S[name][index] /= 1 - self.beta2 ** epoch
        params.append(layer.parameters[index] - self.learning_rate * self.V[name][index] / np.sqrt(
            self.S[name][index] + self.epsilon))
    return params
