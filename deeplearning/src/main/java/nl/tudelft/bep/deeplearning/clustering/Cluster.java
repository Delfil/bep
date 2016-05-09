package nl.tudelft.bep.deeplearning.clustering;

import java.util.ArrayList;
import java.util.Iterator;

public class Cluster implements Comparable<Cluster> {

	private double x, y;
	private int id;
	private ArrayList<Cluster> set;

	public Cluster(double x, double y, int id) {
		this.x = x;
		this.y = y;
		this.id = id;
		this.set = new ArrayList<Cluster>();
	}

	public Cluster(double x, double y, int id, ArrayList<Cluster> clusters) {
		this.x = x;
		this.y = y;
		this.id = id;
		this.set = clusters;
	}

	public Cluster(int cluster) {
		this.x = 0;
		this.y = 0;
		this.id = cluster;
		this.set = new ArrayList<Cluster>();
	}

	@Override
	public int compareTo(Cluster o) {
		return Double.compare(x, o.getX());
	}

	@Override
	public boolean equals(Object other) {
		if (other instanceof Cluster) {
			Cluster o = (Cluster) other;
			if (this.getX() == o.getX() && this.getY() == o.getY() && this.getID() == o.getID()
					&& this.getSet().equals(o.getSet())) {
				return true;
			}
			return false;
		}
		return false;
	}

	public double getX() {
		return x;
	}

	public double getY() {
		return y;
	}

	public int getID() {
		return id;
	}

	public ArrayList<Cluster> getSet() {
		return set;
	}

	public boolean addCluster(Cluster cluster) {
		return set.add(cluster);
	}

	public void calculateMean() {
		double tempX = 0;
		double tempY = 0;
		Iterator<Cluster> iter = set.iterator();
		if (!iter.hasNext()) {
			return;
		}
		while (iter.hasNext()) {
			Cluster temp = iter.next();
			tempX += temp.getX() / set.size();
			tempY += temp.getY() / set.size();
		}
		this.x = tempX;
		this.y = tempY;
	}

	public int size() {
		if (set.size() == 0) {
			return 1;
		} else {
			int res = 0;
			for (Cluster c : set) {
				res += c.size();
			}
			return res;
		}
	}
	
	/**
	 * Gives the indices in the gene activation matrix of the cluster.
	 * @return
	 */
	public ArrayList<Integer> listIndices() {
		if(this.getSet().isEmpty()) {
			ArrayList<Integer> res = new ArrayList<Integer>(1);
			res.add(this.getID());
			return res;
		}
		else {
			ArrayList<Integer> res = new ArrayList<Integer>();
			for(Cluster c : this.getSet()) {
				res.addAll(c.listIndices());
			}
			return res;
		}
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("[mean: " + id + " = (" + x + "," + y + ") \n \t set: {");
		for (Cluster c : this.set) {
			builder.append(c.toString() + ", ");
		}
		builder.append("}]");
		return builder.toString();
	}

	@Override
	public Cluster clone() {
		return new Cluster(this.x, this.y, this.id);
	}
}