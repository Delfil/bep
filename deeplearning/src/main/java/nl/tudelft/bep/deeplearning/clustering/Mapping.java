package nl.tudelft.bep.deeplearning.clustering;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;

import org.nd4j.linalg.io.ClassPathResource;

public class Mapping {
	
	private static int outputNum;

	public static void main(String[] args) throws IOException {
		map("100points.in", "geneact.in", 100, "cluster_1x100");
	}

	/**
	 * Function which takes files containing points and a matrix of gene
	 * activation and outputs a file with correlating genes next to each other
	 * 
	 * @param points string of the file name containing the points
	 * @param geneAct string of the file name containing the gene activation data.
	 * @param imgSize the minimum image we would like to create
	 * @param outputFile the string representing the outputfile name
	 * @throws IOException for when a file isn't found
	 */
	public static void map(String points, String geneAct, int imgSize, String outputFile) throws IOException {
		// Reading file containing points
		Cluster[] layer1 = read(new FileInputStream(new ClassPathResource(points).getFile()));
		int n = layer1.length;
		List<ArrayList<Double>> matrix = readGeneAct(new FileInputStream(new ClassPathResource(geneAct).getFile()), n);
		// Initialize the temporary array representing a layer and run the
		// createCluster for the first time on the input.
		Cluster[] layer2 = new Cluster[n];
		layer2 = createClusters(layer1);

		// Keep looping until one cluster is left.
		while (layer2.length != 1) {
			layer2 = createClusters(layer2);
		}

		int layer = 0;
		while (layerSize(layer2[0], layer) < imgSize) {
			layer++;
		}
		String datFile = outputFile + ".dat";
		String metaFile = outputFile + ".meta";
		List<Cluster> listLayer = layer(layer2[0], layer);
		writeAvgFile(matrix, listLayer, datFile);
		writeMetaFile(listLayer.size(), matrix.size(),true, metaFile);
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
	 */
	public static int layerSize(Cluster root, int layer) {
		if (layer == 0) {
			return 1;
		} else if (root.getList().isEmpty() && layer != 1) {
			throw new RuntimeException();
		} else if (layer > 1) {
			int res = 0;
			for (Cluster c : root.getList()) {
				res += layerSize(c, layer - 1);
			}
			return res;
		} else {
			int res = 0;
			for (int i = 0; i < root.getList().size(); i++) {
				res += 1;
			}
			return res;
		}
	}

	/**
	 * Returns for each layer the clusters in a List. Layer 0 is the root
	 * cluster
	 * 
	 * @param root
	 *            The root cluster
	 * @param layer
	 *            which layer you wish to return
	 * @return Layer as List
	 */
	public static List<Cluster> layer(Cluster root, int layer) {
		if (layer == 0) {
			List<Cluster> res = new ArrayList<Cluster>();
			res.add(root);
			return res;
		} else if (root.getList().isEmpty() && layer != 1) {
			return null;
		} else if (layer > 1) {
			List<Cluster> res = new ArrayList<Cluster>();
			for (Cluster c : root.getList()) {
				res.addAll(layer(c, layer - 1));
			}
			return res;
		} else {
			List<Cluster> res = new ArrayList<Cluster>();
			for (int i = 0; i < root.getList().size(); i++) {
				res.add(root.getList().get(i));
			}
			return res;
		}
	}

	/**
	 * Creates a file with the gene activation data of each patient based on the
	 * indices found by the clustering.
	 * 
	 * @param matrix
	 *            Gene activation of patients
	 * @param indices
	 *            were in the matrix those activations should be
	 * @throws FileNotFoundException
	 *             file not found.
	 * @throws UnsupportedEncodingException
	 *             unsupported encoding,
	 */
	public static void writeFile(List<ArrayList<Double>> matrix, List<Integer> indices, String outputFile)
			throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < indices.size(); j++) {
				writer.print(matrix.get(i).get(indices.get(j)));
				if (j != indices.size() - 1) {
					writer.print(",");
				}
			}
			writer.print("\n");
		}
		writer.close();
	}

	/**
	 * Creates a output file based on which layer is representing the pixels of
	 * the matrix.
	 * 
	 * @param matrix
	 *            the gene activation matrix
	 * @param listLayer
	 *            The layer representing the pixels
	 * @throws FileNotFoundException
	 * @throws UnsupportedEncodingException
	 */
	public static void writeAvgFile(List<ArrayList<Double>> matrix, List<Cluster> listLayer, String outputFile)
			throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < listLayer.size(); j++) {
				Double res = avgActivation(matrix, listLayer.get(j), i);
				writer.print(res);
				if (j != listLayer.size() - 1) {
					writer.print(",");
				}
			}
			writer.print("\n");
		}
		writer.close();
	}
	
	/**
	 * Creates the meta file needed for the neural networks.
	 * @param x number of cells in the matrix
	 * @param numPatients number of patients.
	 * @param one boolean for creating a one by size of layer matrix, otherwise create square matrix.
	 * @param metaOutputFile the name of the output .meta file.
	 * @throws FileNotFoundException if file cannot be created.
	 * @throws UnsupportedEncodingException if encoding isn't supported.
	 */
	public static void writeMetaFile(int x, int numPatients, boolean one, String metaOutputFile) throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(metaOutputFile, "UTF-8");
		writer.println(numPatients);
		if(one) {
			writer.println(1);
			writer.println(x);
		}
		else {
			int res = Double.valueOf(Math.ceil(Math.sqrt(x))).intValue();
			writer.println(res);
			writer.println(res);
		}
		writer.println(outputNum);
		writer.close();
		
	}

	/**
	 * Based on the matrix and the pixel at hand, return the average value of
	 * the cluster
	 * 
	 * @param matrix
	 *            containing all the gene activation data
	 * @param c
	 *            Cluster representing the pixel we want to fill
	 * @param patient
	 *            which patient we wish to pick.
	 * @return the average activation value of the cluster
	 */
	public static Double avgActivation(List<ArrayList<Double>> matrix, Cluster c, int patient) {
		Double res = 0.0;
		for (Integer i : c.listIndices()) {
			res += matrix.get(patient).get(i);
		}
		return res / c.listIndices().size();
	}

	/**
	 * Given a root cluster this function will give a list of the correlating
	 * genes next to each other.
	 * 
	 * @param cluster
	 *            Root cluster of the cluster tree.
	 * @return List with the points sorted for insertion into the matrix.
	 */
	public static ArrayList<Integer> createList(Cluster cluster) {
		ArrayList<Integer> list = new ArrayList<Integer>();

		if (!cluster.getList().isEmpty()) {
			for (Cluster c : cluster.getList()) {
				list.addAll(createList(c));

			}
			return list;
		} else {
			list.add(cluster.getID());
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
		List<Neighbors> neighbors = neighbour(layer1);
		// Temporary ArrayList for the newly created higher level clusters.
		List<Cluster> layer2 = new ArrayList<Cluster>();
		Collections.sort(neighbors);

		int id = 0;
		Cluster cluster = new Cluster(id);
		Neighbors temp = neighbors.remove(0);
		while (!neighbors.isEmpty()) {
			cluster.addCluster(temp.getCluster());
			if (getIndexFromNeighbors(neighbors, temp.getNeighbor()) == -1) {
				layer2.add(cluster);
				id++;
				cluster = new Cluster(id);
				temp = neighbors.remove(0);
			} else {
				temp = neighbors.remove(getIndexFromNeighbors(neighbors, temp.getNeighbor()));
			}
		}
		cluster.addCluster(temp.getCluster());
		layer2.add(cluster);

		// Make from the ArrayList an Array.
		Cluster[] res = layer2.toArray(new Cluster[layer2.size()]);
		return res;
	}

	/**
	 * Helper function which finds the cluster in an array based on its ID.
	 * @param list list of clusters where we would like to find the cluster with ID id
	 * @param id the id we would like to find
	 * @return the index in the array
	 */
	public static int getIndexFromID(Cluster[] list, int id) {
		for (int i = 0; i < list.length; i++) {
			if (list[i].getID() == id) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Helper function which finds the cluster in an list of Neighbors.class.
	 * @param list list of neighbors where we would like to find the cluster c.
	 * @param c the cluster we would like to find.
	 * @return the index of cluster c. 
	 */
	public static int getIndexFromNeighbors(List<Neighbors> list, Cluster c) {
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i).getCluster().getID() == c.getID()) {
				return i;
			}
		}
		return -1;
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

	/**
	 * Reads the gene activation data and returns a matrix represented as a 2 dimensional list
	 * @param in inputstream of the file.
	 * @param numGenes the number of elements each row.
	 * @return matrix representation of the gene activation.
	 */
	public static List<ArrayList<Double>> readGeneAct(InputStream in, int numGenes) {
		List<ArrayList<Double>> matrix = new ArrayList<ArrayList<Double>>();
		Scanner scanner = new Scanner(new InputStreamReader(in));
		outputNum = scanner.nextInt();
		while (scanner.hasNext()) {
			ArrayList<Double> tempPatient = new ArrayList<Double>(numGenes);
			for (int i = 0; i < numGenes; i++) {
				tempPatient.add(scanner.nextDouble());
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
	public static List<Neighbors> neighbour(Cluster[] clusters) {
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
	 * @param Clusters
	 *            Array containing the cluster we would like to know the nearest
	 *            neighbor of.
	 * @param result
	 *            Array where we would like to keep track of the nearest
	 *            neighbor found yet.
	 * @param distances
	 *            Array for keeping track of the distances of the result array.
	 */
	public static void closestClusters(Cluster[] Clusters, int[] result, Double[] distances) {

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
