from core.network import NeuralNetwork

if __name__ == "__main__":
    # Creamos una red de 3 capas: entrada 3 -> oculta 5 -> salida 2
    nn = NeuralNetwork([3, 5, 2], activation="sigmoid")

    nn.summary()

    entrada = [0.1, 0.9, 0.5]
    salida = nn.forward(entrada)
    print(f"Entrada: {entrada}")
    print(f"Salida: {salida}")
