from core.neuron import Neuron

def test_neuron_forward():
    n = Neuron(n_inputs=3, activation="sigmoid")
    output = n.forward([0.2, 0.8, -0.5])
    assert 0 <= output <= 1
