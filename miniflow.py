import numpy as np

class Node(object):
    """
    Base class for nodes in the network

    Arguments:
        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets properties that all nodes need
        """
        # A list of nodes with edges into this node
        self.inbound_nodes = inbound_nodes
        # The eventual value of this node. Set by running the forward()
        # method.
        self.value = None
        # A list of nodes that this node outputs to.
        self.outbound_nodes = []
        # Keys are the inputs to this node and their values are the
        # partials of this node with respect to that input.
        self.gradients = {}
        # Sets this node as an outbound node for all of this node's inputs
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

        # NOTE: Input node is the only node where the value
        # may be passed as an argument to forward().
        #
        # All other node implementations should get the value
        # of the previous node from self.inbound_nodes
        #
        # Example:
        # val0 = self.inbound_nodes[0].value

        def forward(self, value=None):
            # Overwrite the value if one is passed in
            if value is not None:
                self.value = value


class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Add the values that get passed into the Add Class
        """
        self.value = sum([n.value for n in self.inbound_nodes])


class Mul(Node):
    def __init__(self, *input):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Multiply the values that get passed into the Mul Class
        """
        self.value = reduce(lambda x, y: x * y, map((lambda x: x.value), self.inbound_nodes))


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
            # NOTE: The weights and bias properties here are not
            # numbers, but rather references to other nodes.
            # The weight and bias values are stored within the
            # respective nodes.

        def forward(self):
            """
            Set self.value to the value of the Linear function output
            """
            # inputs = self.inbound_nodes[0].value
            # weights = self.inbound_nodes[1].value
            # bias = self.inbound_nodes[2].value
            # self.value = bias
            # for x, w in zip(inputs, weights):
            #    self.value += x * w

            """
            Set the value of this node to the linear transform output
            """
            X = self.inbound_nodes[0].value
            W = self.inbound_nodes[1].value
            b = self.inbound_nodes[2].value
            # Z = X*W + b
            self.value = np.dot(X, W) + b


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        self.x = x
        return 1/(1 + np.exp(-self.x))

    def forward(self):
        """
        Perform the sigmoid function and set the value
        """
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        """
        Calculates the gradient using the derivative of the sigmoid function
        """
        # Initialize the gradients to 0
        self.gradients = {n: np.zeros_like(n.like) for n in self.inbound_nodes}
        # Sum the derivative with respect to the input over all the outputs.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) *grad_cost

class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network
        """
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of
        # shape (3, 1) we get an array of shape (3, 3) as the result when we
        # want an array of shape (3, 1) instead
        #
        # Making both arrays (3, 1) insures the result is (3, 1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.value = np.sum((1/len(y))* np.square(y - a))
        # NOTE: Another method
        # m = self.inbound_nodes[0].value.shape[0]
        # diff = y - a
        # self.value = np.mean(diff**2)


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges)
        `sorted_nodes`: a topologically sorted list of nodes

    Returns output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted nodes.

    Arguments:
    `graph`: The result of calling `topological_sort`
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    for n in graph[::-1]:
        n.backward()
