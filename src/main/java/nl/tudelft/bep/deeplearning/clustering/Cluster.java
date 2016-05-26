package nl.tudelft.bep.deeplearning.clustering;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import nl.tudelft.bep.deeplearning.clustering.exception.MinimumNotPossibleException;

public class Cluster implements Comparable<Cluster> {

	/**
	 * Coordinates of the point or in the case of the cluster the mean.
	 */
	private double x, y;
	private int id;
	private ArrayList<Cluster> list;
	private static final int DOUBLE_BITS = 32;

	public Cluster(final double x, final double y, final int id) {
		this(x, y, id, new ArrayList<Cluster>());
	}

	public Cluster(final double x, final double y, final int id, final ArrayList<Cluster> clusters) {
		this.x = x;
		this.y = y;
		this.id = id;
		this.list = clusters;
	}

	public Cluster(final int cluster) {
		this(0, 0, cluster, new ArrayList<Cluster>());
	}

	@Override
	public int compareTo(final Cluster o) {
		return Double.compare(this.x, o.getX());
	}

	public double getX() {
		return this.x;
	}

	public double getY() {
		return this.y;
	}

	public int getID() {
		return this.id;
	}

	public ArrayList<Cluster> getList() {
		return this.list;
	}

	/**
	 * Adding a cluster to the higher level cluster. Mean is calculated
	 * thereafter.
	 * 
	 * @param cluster
	 *            we wish to add.
	 */
	public void addCluster(final Cluster cluster) {
		this.list.add(cluster);
		this.calculateMean();
	}

	/**
	 * Function for calculating the mean. The mean is save inside the cluster as
	 * x and y.
	 */
	public void calculateMean() {
		double tempX = 0;
		double tempY = 0;
		Iterator<Cluster> iter = this.list.iterator();
		if (!iter.hasNext()) {
			return;
		}
		while (iter.hasNext()) {
			Cluster temp = iter.next();
			tempX += temp.getX() / this.list.size();
			tempY += temp.getY() / this.list.size();
		}
		this.x = tempX;
		this.y = tempY;
	}

	/**
	 * Looks at how many points (leafs of the tree) are in a cluster. If set is
	 * empty its a point, so return 1
	 * 
	 * @return the amount of points in this cluster
	 */
	public int size() {
		if (this.list.size() == 0) {
			return 1;
		} else {
			int res = 0;
			for (Cluster c : this.list) {
				res += c.size();
			}
			return res;
		}
	}

	/**
	 * Gives the indices in the gene activation matrix of the cluster.
	 * 
	 * @return the indices in the gene activation matrix of the cluster
	 */
	public ArrayList<Integer> listIndices() {
		if (this.getList().isEmpty()) {
			ArrayList<Integer> res = new ArrayList<Integer>(1);
			res.add(this.getID());
			return res;
		} else {
			ArrayList<Integer> res = new ArrayList<Integer>();
			for (Cluster c : this.getList()) {
				res.addAll(c.listIndices());
			}
			return res;
		}
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + this.id;
		result = prime * result + ((this.list == null) ? 0 : this.list.hashCode());
		long temp;
		temp = Double.doubleToLongBits(this.x);
		result = prime * result + (int) (temp ^ (temp >>> DOUBLE_BITS));
		temp = Double.doubleToLongBits(this.y);
		result = prime * result + (int) (temp ^ (temp >>> DOUBLE_BITS));
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		Cluster other = (Cluster) obj;
		if (this.id != other.id) {
			return false;
		}
		if (this.list == null) {
			if (other.list != null) {
				return false;
			}
		} else if (!this.list.equals(other.list)) {
			return false;
		}
		if (Double.doubleToLongBits(this.x) != Double.doubleToLongBits(other.x)) {
			return false;
		}
		if (Double.doubleToLongBits(this.y) != Double.doubleToLongBits(other.y)) {
			return false;
		}
		return true;
	}

	/**
	 * Returns for each layer the amount of clusters contained. Layer 0 is the
	 * root cluster
	 * 
	 * @param root
	 *            The root cluster of the tree.
	 * @param layer
	 *            which layer you want the size of.
	 * @return the size of the layer.
	 * @throws MinimumNotPossibleException
	 */
	public int layerSize(final int layer) throws MinimumNotPossibleException {
		if (layer == 0) {
			return 1;
		} else if (this.getList().isEmpty() && layer != 1) {
			throw new MinimumNotPossibleException();
		} else {
			int res = 0;
			for (int i = 0; i < this.getList().size(); i++) {
				res += this.layerSize(layer - 1);
			}
			return res;
		}
	}

	/**
	 * Returns for each layer the clusters in a List. Layer 0 is the root
	 * cluster.
	 * 
	 * @param root
	 *            The root cluster
	 * @param layer
	 *            which layer you wish to return
	 * @return Layer as List
	 * @throws MinimumNotPossibleException
	 */
	public List<Cluster> layer(final int layer) throws MinimumNotPossibleException {
		if (layer == 0) {
			List<Cluster> res = new ArrayList<Cluster>();
			res.add(this);
			return res;
		} else if (this.getList().isEmpty() && layer != 1) {
			throw new MinimumNotPossibleException();
		} else {
			List<Cluster> res = new ArrayList<Cluster>();
			for (Cluster c : this.getList()) {
				res.addAll(c.layer(layer - 1));
			}
			for (Cluster c : res) {
				c.calculateMean();
			}
			return res;
		}
	}
}