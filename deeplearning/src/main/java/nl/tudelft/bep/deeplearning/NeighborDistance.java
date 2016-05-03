package nl.tudelft.bep.deeplearning;

public class NeighborDistance {

	private int[] results;
	private double[] distances;
	
	
	public NeighborDistance(int[] results, double[] distances) {
		this.results = results;
		this.distances = distances;
	}
	
	public int[] getResults() {
		return results;
	}
	
	public double[] getDistances() {
		return distances;
	}
	
	public int getPointIDWithClosestNeighbor() {
		double temp = Double.MAX_VALUE;
		int index = -1;
		for(int i = 0; i < distances.length; i++) {
			if(distances[i] < temp) {
				temp = distances[i];
				index = i;
			}
		}
		return results[index];
	}
	
}
