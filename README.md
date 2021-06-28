# MAXIMIZING THE INFLUENCE SPREAD IN SOCIAL MEDIA USING IMPROVISED CELF ALGORITHM AND CENTRALITY MEASURES



**EXCLUTIVE SUMMARY**

Social networks have attracted a lot of attention as novel information or advertisement diffusion media for viral marketing. Influence maximization describes the problem of finding a small subset of seed nodes in a social network that could maximize the spread of influence. A lot of algorithms have been proposed to solve this problem. Previous algorithms for this were greedy approach, hill climbing algorithm and Cost Effective Lazy Forward (CELF) algorithm. For this project, we will slightly modify the CELF algorithm and we have used the centrality measures property of the nodes, which increases its efficiency and the propagation models are independent cascade and linear threshold. These algorithms are efficient for propagating the influence according to our algorithm and also the execution time is reduced.

Keywords: Influence Maximization (IM), Influence Propagation, Greedy Algorithms, CELF, Sub-modularity, Social Networks.

# **Contents:**

Acknowledgement

[**Abstract:** 3](#_Toc71662439)

[**1.**  **Introduction:** 3](#_Toc71662440)

[**2.**  **Literature Survey:** 4](#_Toc71662441)

[**3.**  **Overview of the Proposed System:** 7](#_Toc71662442)

[**3.1.**  **Introduction:** 7](#_Toc71662443)

[**3.2.**  **Architecture of Proposed System:** 7](#_Toc71662444)

[**3.3.**  **Proposed System Model** 7](#_Toc71662445)

[**4.**  **Proposed System Analysis and Design** 7](#_Toc71662446)

[**5.**  **Implementation** 12](#_Toc71662447)

[**6.**  **Results and Discussion** 12](#_Toc71662448)

[**7.**  **Conclusion, Limitations and Scope for future Work** 18](#_Toc71662449)

[**8.**  **References** 19](#_Toc71662450)

[**Appendix** 19](#_Toc71662451)

1.
# **Introduction:**

  1. OBJECTIVE

Nowadays, mostly everyone in the world are using the social network in one or the other way. So, it is considered as the important media for spreading information, ideas and influence among the individuals. So here comes the concept of influence maximization, because a product developer or an event organizer cannot send the message to each and every node connected in the social network so it is necessary to select some specific users from the network who are able to influence the maximum nodes. These nodes are called influential nodes.

  1. Motivation

In viral marketing strategy, a company invites some initial users, i.e., the seed nodes, to try its new products or technologies. The company would give these initial users free samples and hope that they will give a positive feedback in social networks. By the power of word-of-mouth, these users may affect their neighbors in a social network. These affected neighbors may subsequently propagate the influence to their own neighbors, and so on. The challenge in viral marketing strategy is how to select the seed nodes to maximize return of investment.

Influence maximization in social networks is the problem of selecting a limited size of influential users as seed nodes, and these seed nodes can propagate the message throughout the network. The propagation of information can be fairly quick if the right seed nodes are selected. However, the large scale of social networks and their complicated structures made it challenging to select the right seed nodes. In this paper, we will give a whole set of solution including improving model, designing a novel seed-node selection algorithm and calculating propagation probability.

Hence our concentration is on selecting the influential nodes and designing the propagation model. The most common algorithm for selecting the seed nodes is greedy and hill climbing algorithm. Several following studies have been carried to improve the efficiency of seed-node selection algorithms. Leskovec, Krause and Guestrin proposed a nearly optimal algorithm called Cost Effective Lazy Forward (CELF) algorithm. In this algorithm, the number of nodes to be considered in each round of seed selection is greatly reduced by exploiting the sub-modularity property of models.

This algorithm scaled well to large data sets and their experiments showed that it was 700 times faster than Hill-Climbing algorithm. Our algorithm is optimized approach based on the CELF (Cost Effective Lazy Forward) algorithm. In order to further improve the naive greedy algorithm for influence maximization in social networks, which exploits the property of sub-modularity of the spread function for influence propagation models (e.g., Linear Threshold Model and Independent Cascade Model) to avoid unnecessary steps. Sub- modularity says the marginal gain of a new node shrinks as the set grows. Function f is sub-modular iff f(S {w}) − f(S) ≥ f(T {w}) − f(T) whenever S T. The advantages of this algorithm over existing algorithms are it takes comparatively less time to identify the influential seeds in the social networks.

Since the optimization introduced in CELF is orthogonal to the method used for estimating the spread, our idea can be combined with the heuristic approaches that are based on the greedy algorithm to obtain highly scalable algorithms for influence maximization. Added to these we use the degree centrality and closeness centrality measures of the nodes added to these propagation calculations so that the most influenced seed can be selected.

# **2. Background:**

Influence maximization was first proposed as an algorithm problem by Domingos and Matthew at 2002[1] in his study of viral marketing. Here they have formally defined the problem of maximization of influence and proposed a basic greedy algorithm. If a company&#39;s investment to a user is (I), e.g. sample or advertisement, and the expected return is (R), i.e. when user purchase product from the company, the profit (P) can be determined as P = R – I. Only when P is positive, will a company deem the user as a valuable customer.Calculation of R is different in direct marking vs. viral marketing.

In direct marketing, each user is independent from other users. A company only considers direct purchase action from a user and the user will decide his purchase action independently, not being affected by others&#39; action or persuasion. Therefore, the most valuable user is the user who will purchase most products from the company in direct marketing. Then IM have been classified into two subcategories, one is selecting the most influential seed node and the other is the propagation models.

Kempe, Kleinberg and Tardos provided two different models Independent Cascade model and Linear Threshold model for propagation. These algorithms simulate the influence propagation in social networks. LT model is based on node-specific threshold. The threshold represents the difficulty of switching an inactive node to an active node. A larger threshold value means a node is less likely to switch its status. Then the Independent Cascade model is a dynamic cascade model based on probability theory.

Kempe proposed the first provable approximation of selecting seed nodes by using Hill-climbing algorithm. This algorithm is based on the theory of sub-modular functions. Then comes the CELF algorithm which solves the inefficiency problem of Hill-climbing. This exploits the property of sub modularity of social networks by reducing the candidate nodes. A major limitation of the simple greedy algorithm is twofold: (i) The algorithm requires repeated computes of the spread function for various seed sets. The problem of computing the spread under both IC and LT models is NP-hard.

As a result, Monte-Carlo simulations are run for sufficiently many times to obtain an accurate estimate, resulting in very long computation time. (ii) In each iteration, the simple greedy algorithm searches all the nodes in the graph as a potential candidate for next seed node. As a result, this algorithm entails a quadratic number of steps in terms of the number of nodes. Considerable work has been done on tackling the first issue, by using efficient heuristics for estimating the spread to register huge gains on this front. Relatively little work has been done on improving the quadratic nature of the greedy algorithm.

The most notable work is, where sub-modularity is exploited to develop an efficient algorithm called CELF, based on a &quot;lazy-forward&quot; optimization in selecting seeds. The idea is that the marginal gain of a node in the current iteration cannot be better than its marginal gain in the previous iterations. CELF maintains a table hu, ∆u(S)i sorted on ∆u(S) in decreasing order, where S is the current seed set and ∆u(S) is the marginal gain of u w.r.t S. ∆u(S) is re-evaluated only for the top node at a time and if needed, the table is resorted. If a node remains at the top, it is picked as the next seed. Leskovec empirically shows that CELF dramatically improves the efficiency of the greedy algorithm.

These greedy algorithms were long running, so they proposed a generic algorithm which is an stochastic optimized algorithm like stimulated annealing. This is done though multi-population competition [7]. Then various algorithms like Expansion based method, spatial based indexes, bound based method and hint based method were introduced which uses the nodes locations as well so that the companies can select the proper seeds[8].IMAX query processing was proposed, where we can distinguish within the users. In this method the social network is represent by a graph.

This uses IC model and it is suitable for target aware influence maximization. Time-sensitive algorithm, Time delay, cost are considered in this paper. Because of the monotonicity and sub modularity of this model, a greedy algorithm with (1 - 1/e) approximation ratio is produced. A learning-based approach based on discrete particles warm optimization (LAPSO-IM). Linear threshold and cascade diffusion models are utilized in the approach. This is considered as the tradeoff between quality and efficiency.

Then a greedy algorithm based on local metrics have been proposed to reduce the time complexity of normal greedy algorithm, by creating a mandate vertices set instead of searching the whole vertices set which is done by evaluating the local metrics of each vertex (static and dynamic). Spread-Max consisting of two phases, 1st where the seed nodes are identified using hierarchical reachability approach and the designated seed nodes spread infection during second phase by random walk.

Many hybrid algorithms have evolved after these like, value greed and mountain climbing algorithm which combines traditional greedy and Hill climbing algorithms which eradicates the inefficiencies of those. In the first stage, the region is numerically accumulating rapidly and is easy to activate through value-greed. In the second stage, Hill Climbing Algorithm is run to activate as many nodes as possible on the basis of the first stage.

A hybrid influence maximization that uses both PB-IM (personal based) and CB-IM (community based) is proposed to solve the micro and macro issues of IM- problem. Two strategies were proposed. The PB-CD strategy is used for influence propagation more exactly in community detection. The G-CELF strategy is best for selection of seeds from multiple community accurately. Spanning graph for maximizing the influence spread in Social Networks, a new approach based on the Independent Cascade Model (ICM) which extracts an acyclic spanning graph from the social network. This approach consists to prevent the feedback by eliminating the cycles during the determination of the seeds. Its motivations, its difference from the classical existing approach has been discussed.

# **4. Project description and proposed system**

The following Algorithm describes the CELF algorithm.

- We use σ(S) to denote the spread of seed set S. We maintain a heap Q with nodes corresponding to users in the network G.
- The node of Q corresponding to user u stores a tuple of the form hu.mg1,u.prev best,u.mg2,u.flagi. Here u.mg1 = ∆u(S), the marginal gain of u w.r.t. the current seed set S; u.prev best is the node that has the maximum marginal gain among all the users examined in the current iteration, before user u; u.mg2 = ∆u(S {prev best}), and u.flag is the iteration number when u.mg1 was last updated.
- The idea is that if the node u.prev best is picked as a seed in the current iteration, we don&#39;t need to recompute the marginal gain of u w.r.t (S {prev best}) in the next iteration. It is important to note that in addition to computing ∆u(S), it is not necessary to compute ∆u(S {prev best}) from scratch.
- More precisely, the algorithm can be implemented in an efficient manner such that both ∆u(S) and ∆u(S {prev best}) are evaluated simultaneously in a single iteration of Monte Carlo simulation (which typically contains 10,000 runs). In that sense, the extra overhead is relatively insignificant compared to the huge runtime gains we can achieve if it works out.

Proposed Algorithm:

Input: G,k

Output: seed set S

1. S ← ; Q ← ; last seed = null; cur best = null.
2. for each u V do
3. u.mg1 = σ({u}); u.prev best = cur best; u.mg2 = σ({u,cur best}); u.flag = 0.
4. Add u to Q. Update cur best based on mg1.
5. while |S| \&lt; k do
6. u = top (root) element in Q.
7. if u.flag == |S| then
8. S ← S {u};Q ← Q − {u};last seed = u.
9. continue;
10. else if u.prev best == last seed then
11. u.mg1 = u.mg2.
12. else
13. u.mg1 = ∆u(S); u.prev best = cur best; u.mg2 = ∆u(S {cur best}).
14. u.flag = |S|; Update cur best.
15. Reinsert u into Q and heapify.

- In addition to the data structure Q, the algorithm uses the variables S to denote the current seed set, last seed to track the id of last seed user picked by the algorithm, and cur best to track the user having the maximum marginal gain w.r.t. S over all users examined in the current iteration. The algorithm starts by building the heap Q initially (lines 2-4). Then, it continues to select seeds until the budget k is exhausted.
- As in CELF, we look at the root element u of Q and if u.flag is equal to the size of the seed set, we pick u as the seed as this indicates that u.mg1 is actually ∆u(S) (lines 6-9). The optimization of CELF comes from lines 10- 11 where we update u.mg1 without recomputing the marginal gain. Clearly, this can be done since u.mg2 has already been computed efficiently w.r.t. the last seed node picked.
- If none of the above cases applies, we recompute the marginal gain of u (line 12-13). The propagation model contributes 40% and then the degree centrality measure contributes 20%, closeness centrality measure contributes 40% to the gain of each node.
- We have given more weightage to closeness centrality because if the node has higher closeness centrality measure, it means that it can be able to reach many nodes in shorter time period and degree centrality is used because if the node has higher degree it is connected to many nodes.
- We will be using two propagation models, Independent Cascade (IC) and Linear Threshold (LT) to find the most influential nodes in CELF algorithm.

Propagation Models:

1. **IC Model:**


Algorithm:

Input: graph[v][v],seeds[k]; Output: influnce\_num

1. influences=seeds
2. queue=influences
3. while queue is not null:
4. node= queue.pop()
5. for i in graph[node]:
6. random\_num=random.random()
7. if random\_num\&lt;= graph[node][i]:
8. influences.append(i)
9. influence\_num=length(influences)
10. return influence\_num

1. **LT Model:**


Algorithm:

Input: graph[v][v],seeds[k],in\_degree[n]; Output: influence\_num

1. influence=seeds
2. queue=influences
3. while queue is not null:
4. node=queue.pop()
5. for i in graph[node]:
6. random\_num=random.random()
7. for j in i n\_degree[n]:
8. if j in influences:
9. value+=graph[j][i]
10. if value\&gt;random\_nun:

11. influences.append(i)

12. queue.append(i)

13. influence\_num=length(influences)

14. return influence\_num



# **5. Project demonstration**

Dataset description

Project Code
```python




! pip install networkx

from collections import defaultdict
import random
import time
import networkx as nx

def read_data(graph_file): 
    f1 = open(graph_file, 'r')
    first_line = f1.readline().split() 
    novert = int(first_line[0]) 
    noedge = int(first_line[1]) 
    graph = defaultdict(dict) 
    outdeg = defaultdict(int)
    for line in f1.readlines(): 
        data = line.split() 
        outdeg[int(data[0])] += 1 
        if float(data[2])>0:
            graph[int(data[0])][int(data[1])] ={'weight': float(data[2])} 
        elif float(data[2])<0:
            graph[int(data[0])][int(data[1])] ={'weight': -1*float(data[2])} 
    return novert, noedge, graph, outdeg

#done
def ICpropmodel(graph, seeds): 
    inf = seeds[:]
    qu = inf[:]
    while len(qu) != 0: 
        node = qu.pop(0)
        for element in graph[node]: 
            if element not in inf:
                probility = random.random()
                if probility <= graph[node][element]['weight']: 
                    inf.append(element)
                    qu.append(element) 
    noofinfl = len(inf) 
    return noofinfl

def LTpropmodel(graph, seeds): 
    inf = seeds[:]
    qu = inf[:]
    pre_node_record = defaultdict(float)
    threshold = defaultdict(float)
    while len(qu) != 0: 
        node = qu.pop(0)
        for element in graph[node]: 
            if element not in inf:
                if threshold[element] == 0: 
                    threshold[element] = random.random()
                pre_node_record[element] = pre_node_record[element] + graph[node][element]['weight'] 
                if  pre_node_record[element] >= threshold[element]:
                    inf.append(element) 
                    qu.append(element)
    noofinfl = len(inf) 
    return noofinfl

SN_INFLUENCE_PER = 0.4
CENTRALITY_PER = 0.4
def propic(graph, novert, seed_size, outdeg):
    test_count = 0
    seeds = []
    s_n_influnece = defaultdict(float)
    G=graph
    if G.is_directed():
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.out_degree()}
        G = G.reverse()
    else:
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.degree()}
    path_length = nx.single_source_shortest_path_length
    nodes = G.nodes
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1: 
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            s = (len(sp) - 1.0) / (len(G) - 1)
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    while len(seeds) < seed_size:
        if len(seeds) == 0:
            for node in range(1, novert + 1):
                s_n_influnece[node] = 0
                if node in outdeg.keys():
                    s_n_influnece[node] = (s_n_influnece[node] + ICpropmodel(graph, seeds+[node]))/novert
                    if not closeness_centrality[node]==0:
                        s_n_influnece[node]=s_n_influnece[node]*SN_INFLUENCE_PER+(1/closeness_centrality[node])*CENTRALITY_PER+degreecentrality[node]*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))
                    else:
                        s_n_influnece[node]=s_n_influnece[node]*SN_INFLUENCE_PER+(0)*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))+degreecentrality[node]*CENTRALITY_PER
            max_seed = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece.pop(max_seed)
            seeds.append(max_seed) 
            test_count+=1
        elif len(seeds)!= 0:
            prev_best = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece[prev_best] = 0
            marginal_profit = ICpropmodel(graph, seeds + [prev_best]) - ICpropmodel(graph, seeds)
            s_n_influnece[prev_best] += marginal_profit
            if not closeness_centrality[prev_best]==0:
                s_n_influnece[prev_best]=s_n_influnece[prev_best]*SN_INFLUENCE_PER+(1/closeness_centrality[prev_best])*CENTRALITY_PER+degreecentrality[prev_best]*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))
            else:
                s_n_influnece[prev_best]=s_n_influnece[prev_best]*SN_INFLUENCE_PER+(0)*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))+degreecentrality[prev_best]*CENTRALITY_PER
            current_seed = max(s_n_influnece, key=s_n_influnece.get)
            if current_seed == prev_best:
                seeds.append(current_seed)
                s_n_influnece.pop(current_seed)
            else:
                continue
    return seeds

SN_INFLUENCE_PER = 0.4
CENTRALITY_PER = 0.4
def proplt(graph, novert, seed_size, outdeg):
    seeds = []
    s_n_influnece = defaultdict(float)
    G=graph
    if G.is_directed():
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.out_degree()}
        G = G.reverse()
    else:
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.degree()}
    path_length = nx.single_source_shortest_path_length 
    nodes = G.nodes
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            s = (len(sp) - 1.0) / (len(G) - 1)
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    while len(seeds) < seed_size:
        if len(seeds) == 0:
            for node in range(1, novert + 1):
                s_n_influnece[node] = 0
                if node in outdeg:
                    single_node = []
                    single_node.append(node)
                    s_n_influnece[node]=( s_n_influnece[node] + LTpropmodel(graph, single_node))/novert
                    if not closeness_centrality[node]==0:
                        s_n_influnece[node]=s_n_influnece[node]*SN_INFLUENCE_PER+(1/closeness_centrality[node])*CENTRALITY_PER+degreecentrality[node]*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))
                    else:
                        s_n_influnece[node]=s_n_influnece[node]*SN_INFLUENCE_PER+(0)*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))+degreecentrality[node]*CENTRALITY_PER
            max_seed = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece.pop(max_seed)
            seeds.append(max_seed)
        else:
            prev_best = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece[prev_best] = 0
            new_seeds = seeds + [prev_best]
            marginal_profit = LTpropmodel(graph, new_seeds) - LTpropmodel(graph, seeds)
            s_n_influnece[prev_best] = s_n_influnece[prev_best] + marginal_profit
            if not closeness_centrality[prev_best]==0:
                s_n_influnece[prev_best]=s_n_influnece[prev_best]*SN_INFLUENCE_PER+(1/closeness_centrality[prev_best])*CENTRALITY_PER+degreecentrality[prev_best]*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))
            else:
                s_n_influnece[prev_best]=s_n_influnece[prev_best]*SN_INFLUENCE_PER+(0)*(1-(CENTRALITY_PER+SN_INFLUENCE_PER))+degreecentrality[prev_best]*CENTRALITY_PER
            current_seed = max(s_n_influnece, key=s_n_influnece.get)
            if current_seed == prev_best:
                seeds.append(current_seed)
                s_n_influnece.pop(current_seed)
            else:
                continue
    return seeds

def inffind(graph, novert, seed_size, outdeg, model): 
    if model == "IC":
        seeds = propic(graph, novert, seed_size, outdeg) 
    else:
        seeds = proplt(graph, novert, seed_size, outdeg) 
    return seeds

def calculate_average(graph, seeds, model): 
    if model == "IC":
        count = 0
        total_influence = 0 
        while count < 1000:
            total_influence += ICpropmodel(graph, seeds) 
            count += 1
        IC_average = total_influence/count 
        average_result = IC_average
    else:
        count = 0
        total_influence = 0 
        while count < 1000:
            total_influence += LTpropmodel(graph, seeds) 
            count += 1
        LT_average = total_influence / count 
        average_result = LT_average
    return average_result

def getseeds(G, novert, seed_size, outdeg, model): 
    final_seeds = []
    total_influence = 0
    final_seeds = inffind(G, novert, seed_size, outdeg, model) 
    total_influence = calculate_average(G, final_seeds, model)
    print("Hence the seeds selected for highest influence are\n ", final_seeds)

choice = int(input('1-graph from text file \n 2-random graph \n')) 
if choice==1:
    graph_file=raw_input('enter the name/directory of thefile: ') 
    novert, noedge, graph, outdeg =read_data(graph_file) 
#     print(novert)
#     print(noedge)
#     print(graph.items())
    print(outdeg)
    G = nx.DiGraph()
    G.add_nodes_from(graph)
    G.add_edges_from(((u, v, data)for u, nbrs in graph.items()for v, data in nbrs.items()))
elif choice==2:
    novert=int(input('enter the number of vertices for Erdos-Renghi graph: '))
    prob=float(input('enter the probability for node establishment: '))
    G=nx.fast_gnp_random_graph(novert, prob, seed=None, directed=True)
    noedge=nx.number_of_edges(G)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] =random.random()
    outdeg={}
    for i in G.nodes():
        outdeg[i]=G.out_degree(i)
    print(noedge)
seed_size=int(input('enter the no of seeds required: '))
model=raw_input('Propagation models:\nIC - INDEPENDENT CASCADE\nLT - LINEAR THRESHOLD\nenter: ')
start_time = time.time()
getseeds(G, novert, seed_size, outdeg, model)
print("execution time = ",time.time() - start_time,"s")  
nx.draw_circular(G, node_color = 'bisque', with_labels = True)








```

# **6. Results and Discussion**


**Execution timings:**

This table shows the time taken to execute output to find the highest influence seed sets from the datasets generated randomly using greedy algorithm, celf algorithm and our proposed algorithm.

| **Number of nodes and edges** | **Greedy Algorithm**** (sec) **|** Celf Algorithm (sec) **|** Proposed Algorithm (sec)** |
| --- | --- | --- | --- |
| 50, 1700 | 2.932 | 0.647 | 0.942 |
| 100, 3000 | 12.529 | 2.184 | 3.004 |
| 200, 8000 | 35.635 | 16.85 | 16.87 |
| 300, 18000 | 316.732 | 81.021 | 61.241 |
| 400, 32000 | 649.952 | 237.243 | 157.774 |
| 500, 50000 | 1132.38 | 623.813 | 343.829 |

The below image shows the comparison chart among greedy algorithm, celf algorithm and our proposed algorithm.

![](RackMultipart20210628-4-1yu6iem_html_8594285d091a4d8b.gif)

The above table and plot show that our proposed algorithm is efficient and less time consuming compared to greedy algorithm and celf algorithm as the number of actors increases in the dataset.

# **7. Conclusion, Limitations and Scope for future Work**

In this paper, we proposed the efficient algorithm for solving the influence maximization problem. We have modified CELF algorithm and used centrality measure property to optimize the seed selection process to find the most influential nodes as seed set. Independent cascade and linear threshold propagation models were used. We have used the enhanced algorithm for different real-world datasets and obtained better results. Also, the execution time is reduced while maintaining the efficiency. So clearly this algorithm solves the problem of influence maximization. Our algorithm is likely to be the scalable solution to the influence maximization problem for largescale real-life social networks.

# **8. References**

1.
J. Lee and C. Chung, &quot;A Query Approach for Influence Maximization on Specific Users in Social Networks,&quot; in IEEE Transactions on Knowledge and Data Engineering, vol. 27, no. 2, pp. 340-353, 1 Feb. 2015. DOI: 10.1109/TKDE.2014.2330833
2. Shashank Sheshar Singh, Ajay Kumar, Kuldeep Singh, Bhaskar Biswas.&quot; LAPSO-IM: A learning-based influence maximization approach for social networks &quot;, Department of Computer Science and Engineering, Indian Institute of Technology (BHU), Varanasi, 221–005, India. DOI: https://doi.org/10.1016/j.asoc.2019.105554
3. Huan Li, Ruisheng Zhang, Zhili Zhao, Yongna Yuan, &quot;An Efficient Influence Maximization Algorithm Based on Clique in Social Networks&quot;, IEEE, vol.7, pp. 141083 - 141093, 24 Sep. 2019. DOI: 10.1109/ACCESS.2019.2943412
4. Xiaoheng Deng, Fang Long, Bo Li, Dejuan Cao, Yan Pan, &quot;An Influence Model Based on Heterogeneous Online Social Network for Influence Maximization&quot; in IEEE Transactions on Network Science and Engineering, vol. 7, no. 2, pp. 737-749, 3 June. 2019. DOI: 10.1109/TNSE.2019.2920371
5. Naimisha Kolli, Balakrishnan Narayanaswamy.&quot; Influence Maximization From Cascade Information Traces in Complex Networks in the Absence of Network Structure &quot;, in IEEE Transactions on Computational Social Systems, vol. 6, no. 6, pp. 1147-1155, 19 Sep.2019. DOI: 10.1109/TCSS.2019.2939841
6. Feng Lu, Weikang Zhang, Liwen Shao, Xunfei Jiang, Peng Xu, Hai Jin, &quot;Scalable influence maximization under independent cascade model&quot;, Elsevier Journal of Networks and Computer Applications, vol.86, pp. 15-23, 15 May. 2017. DOI: https://doi.org/10.1016/j.jnca.2016.10.020
7. Matthew Richardson and Pedro Domingos. Mining knowledge-sharing sites for viral marketing. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 61–70. ACM, 2002.
