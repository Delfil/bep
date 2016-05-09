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
	
	public int getPointIDWithClosestNeighbor(int no) {
		Double[] tempDist = distances.clone();
		double temp = Double.MAX_VALUE;
		int index = -1;
		for(int c = 0; c < no; c++) {
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
