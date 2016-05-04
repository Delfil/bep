package nl.tudelft.bep.deeplearning.clustering;

public class NeighborDistance {

	private int[] results;
	private Double[] distances;
	
	
	public NeighborDistance(int[] results, Double[] distances) {
		this.results = results;
		this.distances = distances;
	}
	
	public int[] getResults() {
		return results;
	}
	
	public Double[] getDistances() {
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
