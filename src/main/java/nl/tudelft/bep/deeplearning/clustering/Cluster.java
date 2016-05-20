package nl.tudelft.bep.deeplearning.clustering;

import java.util.ArrayList;
import java.util.Iterator;

public class Cluster implements Comparable<Cluster> {

	//Coordinates of the point or in the case of the cluster the mean.
	private double x, y;
	private int id;
	private ArrayList<Cluster> list;

	public Cluster(double x, double y, int id) {
		this(x, y, id, new ArrayList<Cluster>());
	}

	public Cluster(double x, double y, int id, ArrayList<Cluster> clusters) {
		this.x = x;
		this.y = y;
		this.id = id;
		this.list = clusters;
	}

	public Cluster(int cluster) {
		this(0,0,cluster,new ArrayList<Cluster>());
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
					&& this.getList().equals(o.getList())) {
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

	public ArrayList<Cluster> getList() {
		return list;
	}

	/**
	 * Adding a cluster to the higher level cluster. Mean is calculated thereafter.
	 * @param cluster we wish to add.
	 */
	public void addCluster(Cluster cluster) {
		list.add(cluster);
		this.calculateMean();
	}

	/**
	 * Function for calculating the mean. The mean is save inside the cluster as x and y.
	 */
	public void calculateMean() {
		double tempX = 0;
		double tempY = 0;
		Iterator<Cluster> iter = list.iterator();
		if (!iter.hasNext()) {
			return;
		}
		while (iter.hasNext()) {
			Cluster temp = iter.next();
			tempX += temp.getX() / list.size();
			tempY += temp.getY() / list.size();
		}
		this.x = tempX;
		this.y = tempY;
	}

	/**
	 * Looks at how many points (leafs of the tree) are in a cluster. If set is empty its a point, so return 1
	 * @return the amount of points in this cluster
	 */
	public int size() {
		if (list.size() == 0) {
			return 1;
		} else {
			int res = 0;
			for (Cluster c : list) {
				res += c.size();
			}
			return res;
		}
	}
	
	/**
	 * Gives the indices in the gene activation matrix of the cluster.
	 * @return the indices in the gene activation matrix of the cluster
	 */
	public ArrayList<Integer> listIndices() {
		if(this.getList().isEmpty()) {
			ArrayList<Integer> res = new ArrayList<Integer>(1);
			res.add(this.getID());
			return res;
		}
		else {
			ArrayList<Integer> res = new ArrayList<Integer>();
			for(Cluster c : this.getList()) {
				res.addAll(c.listIndices());
			}
			return res;
		}
	}
}