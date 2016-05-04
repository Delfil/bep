package nl.tudelft.bep.deeplearning;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

import org.nd4j.linalg.io.ClassPathResource;

public class Mapping {

	public static void main(String[] args) throws IOException {

		// Reading file containing points
		Cluster[] layer1 = read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		int n = layer1.length;
		ArrayList<ArrayList<Integer>> matrix = readGeneAct(new FileInputStream(new ClassPathResource("patients.in").getFile()), n);
		// Initialize the temporary array representing a layer and run the
		// createCluster for the first time on the input.
		Cluster[] layer2 = new Cluster[n];
		layer2 = createClusters(layer1);

		// Keep looping until one cluster is left.
		while (layer2.length != 1) {
			layer2 = createClusters(layer2);
		}

		ArrayList<Integer> list = createList(layer2[0]);
		
		
		PrintWriter writer = new PrintWriter("sample_data.dat", "UTF-8");
		for(int i = 0; i < matrix.size(); i++) {
			for(int j = 0; j < n; j++) {
				writer.print(matrix.get(i).get(list.get(j)));
				if(j != n-1) {
					writer.print(",");
				}
			}
			writer.print("\n");
		}
		writer.close();
//		Double dim = Math.ceil(Math.sqrt(layer2[0].size()));
//		int[][] matrix = new int[dim.intValue()][dim.intValue()];
//		int index = 0;
//		for (int i = 0; i < dim.intValue(); i++) {
//			for (int j = 0; j < dim.intValue(); j++) {
//				if (index < list.size()) {
//					matrix[i][j] = list.get(index);
//					index++;
//				}
//				else{
//					matrix[i][j] = -1;
//				}
//			}
//		}
	}

	public static ArrayList<Integer> createList(Cluster cluster) {
		ArrayList<Integer> list = new ArrayList<Integer>();

		if (!cluster.set.isEmpty()) {
			for (Cluster c : cluster.set) {
				list.addAll(createList(c));

			}
			return list;
		} else {
			list.add(cluster.id);
			return list;
		}
	}

	/**
	 * Function which uses the all nearest neighbors algorithm to cluster
	 * clusters together into a higher level cluster. Clusters can also
	 * represent points by having an empty set of clusters.
	 * 
	 * @param layer1
	 *            Array of clusters which we would like to cluster
	 * @return Array containing the newly created higher level clusters.
	 */
	public static Cluster[] createClusters(Cluster[] layer1) {
		// Run nearest neighbor algorithm.
		NeighborDistance result1 = neighbour(layer1.clone());
		// ArrayList for keeping track which cluster is already clustered.
		ArrayList<Boolean> inCluster = new ArrayList<Boolean>(layer1.length);
		for (int i = 0; i < layer1.length; i++) {
			inCluster.add(i, false);
		}
		// Temporary ArrayList for the newly created higher level clusters.
		ArrayList<Cluster> layer2 = new ArrayList<Cluster>();
		// Initialize the index for keeping track of where the cluster at hand
		// is located in layer1
		int index = 0;
		// Keeping track of the id of the new high level clusters.
		int id = 0;
		// Temporary cluster.
		Cluster cluster = new Cluster(id);
		// We keep looping until each cluster is clustered
		while (inCluster.contains(false)) {
			// Case if the cluster isn't in a high level cluster
			if (!inCluster.get(index)) {
				cluster.addCluster(layer1[index]);
				inCluster.set(index, true);
				// Get closest neighbor and find the index in layer1.
				int neighbor = result1.getResults()[index];
				for (int i = 0; i < layer1.length; i++) {
					if (layer1[i].getID() == neighbor) {
						index = i;
						break;
					}
				}

			}
			// Case if the closest neighbor is already in a high level cluster.
			else {
				cluster.calculateMean();
				layer2.add(cluster);
				id++;
				// Create new high level cluster and find the next cluster not
				// in a high level cluster.
				cluster = new Cluster(id);
				index = inCluster.indexOf(false);
			}
		}
		cluster.calculateMean();
		layer2.add(cluster);
		// Make from the ArrayList an Array.
		Cluster[] res = new Cluster[layer2.size()];
		int i = 0;
		for (Cluster c : layer2) {
			res[i] = c;
			i++;
		}
		return res;
	}

	/**
	 * Reads an .in file with the number of points followed by the coordinates
	 * of these points.
	 * 
	 * @param in
	 *            Inputstream of the file
	 * @return Array with the points represented by Cluster.class.
	 */
	public static Cluster[] read(InputStream in) {
		Scanner scanner = new Scanner(new InputStreamReader(in));
		int n = scanner.nextInt();

		Cluster[] clusters = new Cluster[n];

		for (int i = 0; i < n; i++) {
			clusters[i] = (new Cluster(scanner.nextFloat(), scanner.nextFloat(), i));
		}
		scanner.close();

		return clusters;

	}
	
	public static ArrayList<ArrayList<Integer>> readGeneAct(InputStream in, int numGenes) {
		ArrayList<ArrayList<Integer>> matrix = new ArrayList<ArrayList<Integer>>();
		Scanner scanner = new Scanner(new InputStreamReader(in));
		
		while(scanner.hasNext()) {
			ArrayList<Integer> tempPatient = new ArrayList<Integer>(numGenes);
			for(int i = 0; i < numGenes; i++) {
				tempPatient.add(scanner.nextInt());
			}
			matrix.add(tempPatient);
		}
		scanner.close();
		return matrix;
	}

	/**
	 * Returns the nearest neighbor for each cluster or point as an array.
	 * 
	 * @param clusters
	 *            The clusters we would like to know the nearest neighbors of.
	 * @return Array of nearest neighbors
	 */
	public static NeighborDistance neighbour(Cluster[] clusters) {
		int[] result = new int[clusters.length];
		// Distances is used to keep track of the closest distance found for
		// each point.
		double[] distances = new double[clusters.length];
		for (int i = 0; i < clusters.length; i++) {
			distances[i] = Double.MAX_VALUE;
		}
		closestClusters(clusters, result, distances);
		return new NeighborDistance(result, distances);
	}

	/**
	 * Recursive function for finding the nearest neighbor for each
	 * cluster/point.
	 * 
	 * @param Clusters
	 *            Array containing the cluster we would like to know the nearest
	 *            neighbor of.
	 * @param result
	 *            Array where we would like to keep track of the nearest
	 *            neighbor found yet.
	 * @param distances
	 *            Array for keeping track of the distances of the result array.
	 */
	public static void closestClusters(Cluster[] Clusters, int[] result, double[] distances) {

		int n = Clusters.length;

		if (n == 1) {
			return;
		}

		java.util.Arrays.sort(Clusters);

		int mid = Clusters.length / 2;
		Cluster median = Clusters[mid];
		Cluster[] part1 = new Cluster[mid];
		Cluster[] part2 = new Cluster[n - mid];

		part1 = Arrays.copyOfRange(Clusters, 0, mid);
		part2 = Arrays.copyOfRange(Clusters, mid, n);

		closestClusters(part1, result, distances);
		closestClusters(part2, result, distances);

		ArrayList<Cluster> strip = new ArrayList<Cluster>();

		double max = 0;
		for (int i = 0; i < n; i++) {
			double distance = distances[Clusters[i].getID()];
			if (distance > max) {
				max = distance;
			}
		}

		for (int i = 0; i < n; i++) {
			if (Math.abs(median.getX() - Clusters[i].getX()) < max) {
				strip.add(Clusters[i]);
			}
		}

		Collections.sort(strip, new Comparator<Cluster>() {
			public int compare(Cluster p, Cluster p2) {
				return Double.compare(p.getY(), p2.getY());
			}
		});

		for (int i = 0; i < strip.size(); i++) {
			int k = strip.size() - i - 1;
			if (k > 8) {
				k = 9 + i;
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
