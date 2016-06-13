package nl.tudelft.bep.deeplearning.clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public final class NearestNeighbors {
	
	private static final int NEIGHBORS_STRIP = 8;
	
	/**
	 * Utility-classes should not be initialized.
	 */
	private NearestNeighbors() {
	}

	/**
	 * Helper function which finds the cluster in an array based on its ID.
	 * 
	 * @param list
	 *            list of clusters where we would like to find the cluster with
	 *            ID id
	 * @param id
	 *            the id we would like to find
	 * @return the index in the array
	 */
	public static int getIndexFromID(final Cluster[] list, final int id) {
		for (int i = 0; i < list.length; i++) {
			if (list[i].getID() == id) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Returns the nearest neighbor for each cluster or point as an array.
	 * 
	 * @param clusters
	 *            The clusters we would like to know the nearest neighbors of.
	 * @return Array of nearest neighbors
	 */
	public static List<Neighbors> neighbour(final Cluster[] clusters) {
		int[] result = new int[clusters.length];
		// Distances is used to keep track of the closest distance found for
		// each point.
		Double[] distances = new Double[clusters.length];
		for (int i = 0; i < clusters.length; i++) {
			distances[i] = Double.MAX_VALUE;
		}
		closestClusters(clusters.clone(), result, distances);
		List<Neighbors> neighbors = new ArrayList<Neighbors>();
		for (int i = 0; i < clusters.length; i++) {
			int index = getIndexFromID(clusters, result[i]);
			neighbors.add(new Neighbors(clusters[i], clusters[index], distances[i]));
		}
		return neighbors;
	}

	/**
	 * Recursive function for finding the nearest neighbor for each
	 * cluster/point.
	 * 
	 * @param clusters
	 *            Array containing the cluster we would like to know the nearest
	 *            neighbor of.
	 * @param result
	 *            Array where we would like to keep track of the nearest
	 *            neighbor found yet.
	 * @param distances
	 *            Array for keeping track of the distances of the result array.
	 */
	public static void closestClusters(final Cluster[] clusters, final int[] result, final Double[] distances) {

		int n = clusters.length;

		if (n == 1) {
			return;
		}

		java.util.Arrays.sort(clusters);

		int mid = clusters.length / 2;
		Cluster median = clusters[mid];
		Cluster[] part1 = new Cluster[mid];
		Cluster[] part2 = new Cluster[n - mid];

		part1 = Arrays.copyOfRange(clusters, 0, mid);
		part2 = Arrays.copyOfRange(clusters, mid, n);

		closestClusters(part1, result, distances);
		closestClusters(part2, result, distances);

		Double max = maxDistance(clusters, distances);

		ArrayList<Cluster> strip = createStrip(clusters, max, median.getX());
		calculateDistancesStrip(strip, distances, result);

	}

	/**
	 * Returns the maximum distance found in the recursive calls so far.
	 * 
	 * @param clusters
	 *            all the points/clusters we wish to find the shortest distance
	 *            found of.
	 * @param distances
	 *            array containing the distances.
	 * @return maximum distance found in the recursive calls so far.
	 */
	private static double maxDistance(Cluster[] clusters, Double[] distances) {
		double max = 0;
		for (int i = 0; i < clusters.length; i++) {
			double distance = distances[clusters[i].getID()];
			if (distance > max) {
				max = distance;
			}
		}
		return max;
	}

	/**
	 * Creates the strip with points closer to the median than the maximum
	 * distance found in the two parts of points.
	 * 
	 * @param clusters
	 *            array containing the all the points/clusters.
	 * @param max
	 *            greatest distance between points found in the two parts
	 * @param medianX
	 *            the x-coordinate of the median.
	 * @return
	 */
	private static ArrayList<Cluster> createStrip(Cluster[] clusters, Double max, Double medianX) {
		ArrayList<Cluster> strip = new ArrayList<Cluster>();
		for (int i = 0; i < clusters.length; i++) {
			if (Math.abs(medianX - clusters[i].getX()) < max) {
				strip.add(clusters[i]);
			}
		}

		Collections.sort(strip, new Comparator<Cluster>() {
			public int compare(final Cluster p, final Cluster p2) {
				return Double.compare(p.getY(), p2.getY());
			}
		});
		return strip;
	}

	/**
	 * Calculates the distances between points in the strip and updates results
	 * if a smaller distance is found.
	 * 
	 * @param strip
	 *            containing all the points in the strip
	 * @param distances
	 *            total array keeping track of the shortest distances found so
	 *            far.
	 * @param result
	 *            total array keeping track of the ID of the closest point found
	 *            so far.
	 */
	private static void calculateDistancesStrip(ArrayList<Cluster> strip, Double[] distances, int[] result) {
		for (int i = 0; i < strip.size(); i++) {
			int k = strip.size() - i - 1;
			if (k > NEIGHBORS_STRIP) {
				k = NEIGHBORS_STRIP + 1 + i;
			} else {
				k = strip.size();
			}
			for (int j = i + 1; j < k; j++) {
				double tempDistance = Math.sqrt(Math.pow(strip.get(j).getX() - strip.get(i).getX(), 2)
						+ Math.pow(strip.get(j).getY() - strip.get(i).getY(), 2));
				if (distances[strip.get(i).getID()] > tempDistance) {
					result[strip.get(i).getID()] = strip.get(j).getID();
					distances[strip.get(i).getID()] = tempDistance;
				}
				if (distances[strip.get(j).getID()] > tempDistance) {
					result[strip.get(j).getID()] = strip.get(i).getID();
					distances[strip.get(j).getID()] = tempDistance;
				}
			}
		}
	}

}
