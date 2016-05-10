package nl.tudelft.bep.deeplearning.clustering;

public class Neighbors implements Comparable<Neighbors>  {

	private Cluster c;
	private Cluster neighbor;
	private Double distance;
	
	public Neighbors(Cluster c, Cluster neighbor, Double distance) {
		this.c = c;
		this.neighbor = neighbor;
		this.distance = distance;
	}

	@Override
	public int compareTo(Neighbors o) {
		return this.distance.compareTo(o.getDistance());
	}
	
	public Double getDistance() {
		return this.distance;
	}
	
	public Cluster getCluster() {
		return this.c;
	}
	
	public Cluster getNeighbor() {
		return this.neighbor;
	}
}
