package nl.tudelft.bep.deeplearning;

import java.util.Iterator;
import java.util.Set;

public class Cluster implements Comparable<Cluster> {

	double x, y;
	int id;
	Set<Cluster> set;

	public Cluster(double x, double y, int id) {
		this.x = x;
		this.y = y;
		this.id = id;
	}

	public Cluster(double x, double y, int id, Set<Cluster> clusters) {
		this.x = x;
		this.y = y;
		this.id = id;
		this.set = clusters;
	}

	@Override
	public int compareTo(Cluster o) {
		return Double.compare(x, o.getX());
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

	public boolean addCluster(Cluster cluster) {
		return set.add(cluster);
	}

	public void calculateMean() {
		double tempX = 0;
		double tempY = 0;
		Iterator<Cluster> iter = set.iterator();
		if(!iter.hasNext()) {
			return;
		}
		while (iter.hasNext()) {
			Cluster temp = iter.next();
			tempX += temp.getX()/temp.size();
			tempY += temp.getY()/temp.size();
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

	@Override
	public String toString() {
		return id + " = (" + x + "," + y + ")";
	}

	@Override
	public Cluster clone() {
		return new Cluster(this.x, this.y, this.id);
	}
}