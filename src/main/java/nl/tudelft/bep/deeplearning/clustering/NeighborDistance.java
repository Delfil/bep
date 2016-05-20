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
	
	/**
	 * Gets the nth point where the points are sorted on the distance to its nearest neighbor
	 * @param n the nth point.
	 * @return the nth point's ID where the points are sorted on distance to neighbor.
	 */
	public int getPointIDWithClosestNeighbor(int n) {
		Double[] tempDist = distances.clone();
		double temp = Double.MAX_VALUE;
		int index = -1;
		for(int c = 0; c < n; c++) {
			index = -1;
			temp = Double.MAX_VALUE;
			for(int i = 0; i < tempDist.length; i++) {
				if(tempDist[i] < temp) {
					temp = tempDist[i];
					index = i;
				}
			}
			tempDist[index] = Double.MAX_VALUE;
		}
		
		return results[index];
	}
	
}
