from core.network import NeuralNetwork

def test_forward_pass():
    nn = NeuralNetwork([3, 5, 2], activation="tanh")
    output = nn.forward([0.1, 0.5, -0.3])
    assert len(output) == 2
    for val in output:
        assert -1 <= val <= 1
