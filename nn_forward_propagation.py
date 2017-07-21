"""
This script builds and runs a graph with miniflow
"""

from miniflow import *

x, y, z = Input(), Input(), Input()

add_f = Add(x, y, z)
mul_f = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
add_output = forward_pass(add_f, graph)
mul_output = forward_pass(mul_f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], add_output))
print("{} x {} x {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], mul_output))
