# CMU 15-618 Parallel Computer Architecture and Programming Project  

# Parallel Influence Maximization in Social Networks  

Yuling Wu (yulingw), Zican Yang (zicany)

## SUMMARY  
We are going to implement a parallel version of influence maximization algorithm in social networks using OpenMP.

## BACKGROUND  
Social networks have become an important part of people's lives due to the rise of mobile devices and the continuous development of network technology. The user relationships on many platforms can be represented by social networks. How to better analyze social networks and use their characteristics has become the focus of research. For example, in the fields of public opinion monitoring and viral marketing, the goal is to find a fixed number of seed user nodes to maximize the spread of influence. Such goals can be abstracted as Influence Maximization.

There are three common models in influence spread, and we choose independent cascade(IC) model. Suppose we have a directed Graph *G=(V,E,W)* to represent a social network, where each vertex in *V* is a social network user, each edge in *E* is an influence spread route from a user to another, and each value in *W* is the weight of an edge which represents the probability of influence spread. The influence maximization problem can be depicted as, given the number *N* of seed nodes, select *N* seed nodes from *V* which will result in maximum number of activated nodes.

Because we mainly focus on the parallelizing of the problem, we will make every assumption simpler. We suppose that our graph is a undirected graph, and each value in *W* is identical (we will make influence spread possibility as a command line argument). We also suppose that each vertex can only perform influence spread once, which means if a vertex is visited, it cannot be revisited.

This problem is proved NP-hard<sup>[[3]](#Reference)</sup>, and there are two kinds of mainstream algorithms to solve it. One is greedy algorithms<sup>[[3]](#Reference)</sup>, another is heuristic algorithms<sup>[[4]](#Reference)</sup>. 

The basic greedy algorithm is, we compute the spread result on each possible assignment of *N* seed nodes on *V*, and select the set of nodes with the maximum spread result. This can be parallized in two levels: the first is we can assign the possible assignment of seed nodes to *P* processors, and combine the result from each precessor's local maximum value; the second is we can parallize the computation of influence spread given a fixed set of seed nodes by computing each node's spread and combine them together.

The basic heuristic algorithm is, we compute the out-degree of each vertice and sort in descending order. Then we select the nodes with top-N out-degree as seed nodes set. To make the result more precise, we may need minus-1 degree or degree-discount algorithm<sup>[[4]](#Reference)</sup> to modify the out-degrees to avoid selecting nodes too close to each other. This can be parallized in two levels: the first is computing the out-degree of each vertice in parallel; the second is selecting top-N nodes from observing sorted out-degrees.

## THE CHALLENGE  

#### Challenge  
The discussion of influence spread proposed by Pedro Domingos and Matt Richardson<sup>[[1][2]](#Reference)</sup> is a type of [NP Hardness](https://en.wikipedia.org/wiki/NP-hardness) question. In the Greedy and Heuristic algorithms we discuss above that performed by David Kempe<sup>[[3]](#Reference)</sup> and Wei Chen<sup>[[4]](#Reference)</sup>, they are serialization algorithm which need lots of time to finish their calculation. The reason is they are several internal dependencies among the data so that parallelization will be a challenging task. Hence, the key challenge point is how do we find out a suitable way to eliminate the dependencies and how do we balance the trade-off between correctness and performance.

#### What to learn
1. Deeply understand and grasp the usage of OpenMP.  
2. Further build a good sense of parallelization and the trade-off behind correctness and performance.  
3. Grasp and understand the Influence Maximization problem in Social Networks.  
4. Understand the art of parallelization.  

#### Workload  
- The dependency is that each node's spread direction and possibility is affected by the previous/neighbor node's status.  
- If paralleled, the computing units (processes) may need to access the same nodes in the Network. The communication will be high.
- Under our preliminary design, all the computing units (processes) will do the same execution.

#### Constraints  
- The CPU Frequency and data transfer rate limit the calculation speed.  
- Decoupling the dependency will lead to inaccurate final results.  

## RESOURCES  
We plan to start from scratch and implement a sequential version of the algorithm first. We have not yet decided to use greedy algorithm or heuristic algorithm, but they can all be parallelized. Because implementing a sequential version of both algorithm will not be too difficult, we may implement sequential of them both.

We will reference some research papers in the *Reference* part to implement the sequential version of algorithms.

We will use the ghc machines to develop and PSC machines to test the performance. The PSC machines have more cores and can let us test the performance with higher parallization.

## GOALS AND DELIVERABLES  
#### PLAN TO ACHIEVE (100%)
1. Use OpenMP to parallelize the sequential influence spread algorithm, and expect a speedup of close to P-1 when *P* is not so high.  
2. Test the parallel algorithm on different datasets to ensure the performance speedup can scale.  
3. Verify a drop in speedup when *P* is very high due to extra work.
4. Verify that the spread of influence will not drop too much.
5. Test and analyse the performance on both the serialize algorithm and the parallel algorithm from different input args (number of seeds, spread probability, etc.).  
6. Build a user-defined tool to generate arbitrary kind of dataset which can be used to test the two kind of algorithms.

#### HOPE TO ACHIEVE  (125%)  
1. Our speedup will be close to P when *P* is not so high.
2. The spread of influence will be very close to the serial algorithm.
3. Produce a multi-dimensional scaling report that analyze the features from serial and parallel algorithm on different influence maximization algorithms(greedy, basic heuristic, minus one heuristic, degree discount heuristic).

#### IN CASE THE WORK GOES MORE SLOWLY (75%)
1. If the algorithm will not work so well, we hope in the best situation it will give us a 2x speedup.  
2. ~~Finish serialization and parallel versions in only one of the algorithms (Greedy or Heuristic).~~ (Already finished before milestone)

#### DEMO PLAN
We plan to give visualization of our influence spread result, graph of speedup and result analysis in our final poster session)

## PLATFORM CHOICE
We choose C/C++ to implement our solution. First, we have large datasets, and C/C++ will make the program runs much faster than Python. Second, OpenMP gives good support to C/C++ implementation.

We choose OpenMP to parallize our algorithm. To solve the dependency problem in original sequential algorithm, shared-memory model will be good for us to start with. We can use some locks to solve contention problems.

## SCHEDULE  
Week 1: Implement sequential greedy and heuristic algorithms and choose which one to further parallelize.  

Week 2: Use OpenMP to develop parallel version of the algorithm.  

Week 3: Finish developing algorithm and try to optimize its performance.  

Week 4: Test on machines, complete report and prepare for demo and presentation.  

## Reference  

[1] Domingos P, Richardson M (2001) Mining the network value of customers. In: Proc
SIGKDD, San Francisco, pp 57–66

[2] Richardson M, Domingos P (2002) Mining knowledge-sharing sites for viral
marketing. In: Proc SIGKDD, Edmonton, Alberta, pp 61–70

[3] Kempe D, Kleinberg J, Tardos E (2003) Maximizing the spread of influence through
a social network. In: Proc SIGKDD, Washington, pp 137–146

[4] Chen W , Wang Y , Yang S . Efficient influence maximization in social networks[C]. Proceedings of the 15th ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining, Paris, France, June 28 - July 1, 2009. ACM, 2009.

## Milestone
In one to two paragraphs, summarize the work that you have completed so far. (This should be easy if you have been maintaining this information on your project page.)

- We have implemented both the serial version of greedy algorithm and heuristic algorithm, and we have implemented basic heuristic, minus one heuristic, and degree discount heuristic. We have imported real-world social network datasets, and implemented a databuilder for manually building some data for test.
- We have tested the serial algorithm on datasets, and found out that the heuristic algorithm runs much faster than greedy algorithm, but has a worth spread result. We also found out that increasing spread probability will give a increasing spread result, but will give a first increasing then decreasing compute time of the program. To make testing and analyzing our future parallel implementation easier, we should choose a proper spread probability. Also, we should test greedy algorithm only on small datasets because it will take very long time to compute on large datasets. 

Describe how you are doing with respect to the goals and deliverables stated in your proposal. Do you still believe you will be able to produce all your deliverables? If not, why? In your milestone writeup we want an updated list of goals that you plan to hit for the poster session.

- We have build the full test code on the serialization version on both Greedy and Heuristic approaches.
- Designed and implemented a dataset generator to build any kind of networks.
- Tested on typical networks and collected performance results.

Yes, we still believe we can produce all the deliverables. At the meantime, we have updated the goals and plans above.

What do you plan to show at the poster session? Will it be a demo? Will it be a graph?

- Basic introduction of the prject background and question we want to solve.  
- Several typical Social Network Graphs as well as one of our test data.  
- A Demo about how to run our experiments.
- The results and comparison plots based on our experiments.  
- The challenges we have met in this project and the solutions/efforts we performed.  

Do you have preliminary results at this time? If so, it would be great to included them in your milestone write-up.

- First, we tested on serial greedy algorithm on our manually built easy.txt dataset which contains 40 vertices and 600 edges, with number of seeds = 5 and different spread probability to see the spread result and compute time. We can see that as the spread probability increases, the spread result increses, but the compute time first increases, then decreases.  

|               | p = 0.01 | p = 0.05 | p = 0.1 | p = 0.2 | p= 0.5 | p = 1  |
|---------------|----------|----------|---------|---------|--------|--------|
| Spread Result | 7        | 29       | 40      | 40      | 40     | 40     |
| Compute Time  | 118.93   | 396.60   | 593.92  | 711.19  | 959.37 | 551.30 |  

- Second, we tested on three kinds of heuristic algorithms on facebook datasets which contains 4039 vertices and 88234 edges with default settings and different spread probability.
Basic Heuristic:  

|               | p = 0.01 | p = 0.05 | p = 0.1 | p = 0.2 | p= 0.5 | p = 1 |
|---------------|----------|----------|---------|---------|--------|-------|
| Spread Result | 312      | 2156     | 2963    | 3512    | 3935   | 4039  |
| Compute Time  | 0.035    | 0.110    | 0.133   | 0.160   | 0.216  | 0.129 |  

MinusOne Heuristic:  

|               | p = 0.01 | p = 0.05 | p = 0.1 | p = 0.2 | p= 0.5 | p = 1 |
|---------------|----------|----------|---------|---------|--------|-------|
| Spread Result | 302      | 2169     | 2955    | 3512    | 3935   | 4039  |
| Compute Time  | 0.034    | 0.107    | 0.130   | 0.158   | 0.212  | 0.118 |

DegreeDiscount Heuristic:  

|               | p = 0.01 | p = 0.05 | p = 0.1 | p = 0.2 | p= 0.5 | p = 1 |
|---------------|----------|----------|---------|---------|--------|-------|
| Spread Result | 311      | 2165     | 2955    | 3578    | 3939   | 4039  |
| Compute Time  | 0.039    | 0.113    | 0.135   | 0.162   | 0.216  | 0.124 |  

List the issues that concern you the most. Are there any remaining unknowns (things you simply don't know how to solve, or resource you don't know how to get) or is it just a matter of coding and doing the work? If you do not wish to put this information on a public web site you are welcome to email the staff directly.  

1. The serialization version of Greedy algorithm runs too slow to get the result in a short time, especially when the dataset is large.
2. No enough official Social Network dataset for wide-range validation on our algorithms.
3. Parallel design still have yet to be determined because we can parallelize the algorithm among monteCarloSimulation level, singleNodeBFS level, and verticesSelection level. More incoming demos may be helpful to make the final decision on the parallel design.