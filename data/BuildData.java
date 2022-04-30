/**
 * CMU 15-618 Parallel Computer Architecture and Programming
 * Parallel Influence Maximization in Social Networks
 * 
 * Author: Zican Yang(zicany), Yuling Wu(yulingw)
 */

import java.util.*;

/**
 * Social Network Graph Generation Tool
 *
 * Use this tool to generate any kind of testset you want
 */
public class BuildData {
    // Number of vertices
    static int vecNum = 20;
    // Number of total edges
    static int edgeNum = 200;

    // Edge start offset
    static int base = 0;

    private static class Edge {
        int srcVec;
        int destVec;

        public Edge(int srcVec, int destVec) {
            this.srcVec = srcVec;
            this.destVec = destVec;
        }
    }

    public static void main(String[] args) {

        Random rand = new Random(System.currentTimeMillis());

        HashMap<Integer, HashSet<Integer>> record = new HashMap<>();

        // Edge should be listed in ascending order
        PriorityQueue<Edge> result = new PriorityQueue<>(Comparator.comparingInt((Edge o) -> o.srcVec));

        int currEdgeNum = 0;
        // Edge should fit in range
        if (edgeNum > (vecNum * (vecNum - 1))/2) {
            edgeNum = (vecNum * (vecNum - 1))/2;
        }

        while (currEdgeNum < edgeNum) {
            int srcVec = rand.nextInt(vecNum);
            int destVec = rand.nextInt(vecNum);

            if (srcVec == destVec) {
                continue;
            } else if (record.containsKey(srcVec) && record.get(srcVec).contains(destVec)) {
                continue;
            } else if (record.containsKey(destVec) && record.get(destVec).contains(srcVec)) {
                continue;
            }

            result.add(new Edge(srcVec, destVec));

            HashSet<Integer> currSet = record.getOrDefault(srcVec, new HashSet<>());
            currSet.add(destVec);
            record.put(srcVec, currSet);

            currEdgeNum++;
        }

        System.out.println(vecNum + " " + edgeNum);

        while(!result.isEmpty()) {
            Edge e = result.poll();
            System.out.println((e.srcVec+base) + " " + (e.destVec+base));
        }
    }
}