import numpy as np
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Properties will go here!

        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes

        # Node(s) to which this Node passes values
        self.outbound_nodes = []

        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        # A calculated value, initialized to None
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on 'inbound_nodes' and store the result in self.value.
        """
        raise NotImplemented

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
        self.value = self._sigmoid(self.inbound_nodes[0].value)

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
