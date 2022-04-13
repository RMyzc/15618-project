import java.util.*;

public class BuildData {
    static int vecNum = 20;
    static int edgeNum = 200;

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

        PriorityQueue<Edge> result = new PriorityQueue<>(Comparator.comparingInt((Edge o) -> o.srcVec));

        int currEdgeNum = 0;
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