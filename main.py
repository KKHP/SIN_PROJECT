from collections import defaultdict import random
import time
import networkx as nx
import myReadData as read
import myICpropmodel as IC
import myLTpropmodel as LT



seed_size=int(input('enter the no of seeds required: ')) 
model=input('Propagation models:\nIC - INDEPENDENT CASCADE\nLT - LINEAR THRESHOLD\nenter:	')
start_time = time.time()
getseeds(G, novert, seed_size, outdeg, model) 
print("execution time = ",time.time() - start_time,"ms")