from model import *

m = Model("labels.csv")
print(m.get_y().shape)
print(m.get_X().shape)
