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
	
	private Double getDistance() {
		return this.distance;
	}
	
	protected Cluster getCluster() {
		return this.c;
	}
	
	protected Cluster getNeighbor() {
		return this.neighbor;
	}
}
