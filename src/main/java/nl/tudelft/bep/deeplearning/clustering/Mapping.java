package nl.tudelft.bep.deeplearning.clustering;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import nl.tudelft.bep.deeplearning.clustering.exception.MinimumNotPossibleException;

public final class Mapping {

	private static final int BATCH_SIZE = 32;
	private static final double TRAINING_SET = 0.7;
	private static int outputNum;
	private static final Logger LOG = LoggerFactory.getLogger(Mapping.class);

	public static void main(final String[] args) throws IOException {
		String pointFile = args[0];
		String geneAct = args[1];
		String datFile = args[2] + ".dat";
		String metaFile = args[2] + ".meta";
		String min = args[3];
		Integer minimum = Integer.valueOf(min);
		String dim = args[4];
		Boolean dimensions = Boolean.valueOf(dim);

		Cluster[] points = read(new FileInputStream(new ClassPathResource(pointFile).getFile()));
		List<ArrayList<Double>> matrix = readGeneAct(new FileInputStream(new ClassPathResource(geneAct).getFile()),
				points.length);

		List<Cluster> listLayer = map(points, matrix, minimum);

		writeFile(matrix, listLayer, datFile);
		writeMetaFile(listLayer.size(), matrix.size(), dimensions, metaFile);
	}

	/**
	 * This utility class should not be constructed.
	 */
	private Mapping() {
	}

	/**
	 * Function which takes files containing points and a matrix of gene
	 * activation and returns a list of the cluster we would like to put inside
	 * the matrix.
	 * 
	 * @param points
	 *            string of the file name containing the points
	 * @param geneAct
	 *            string of the file name containing the gene activation data.
	 * @param imgSize
	 *            the minimum image we would like to create
	 * @return A list of the cluster we would like to put inside the matrix.
	 */
	public static List<Cluster> map(final Cluster[] points, final List<ArrayList<Double>> geneAct, final int imgSize) {
		int n = points.length;
		// Initialize the temporary array representing a layer and run the
		// createCluster for the first time on the input.
		Cluster[] layer2 = new Cluster[n];
		layer2 = createClusters(points);

		// Keep looping until one cluster is left.
		while (layer2.length != 1) {
			layer2 = createClusters(layer2);
		}

		int layer = 0;

		try {
			while (layer2[0].layerSize(layer) < imgSize) {
				layer++;
			}
			return layer2[0].layer(layer);
		} catch (MinimumNotPossibleException e) {
			LOG.info("Minimum not possible, all points are selected");
			return createList(layer2[0]);
		}

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
	public static void writeFile(final List<ArrayList<Double>> matrix, final List<Cluster> listLayer,
			final String outputFile) throws FileNotFoundException, UnsupportedEncodingException {
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
	 * 
	 * @param x
	 *            number of cells in the matrix
	 * @param numPatients
	 *            number of patients.
	 * @param one
	 *            boolean for creating a one by size of layer matrix, otherwise
	 *            create square matrix.
	 * @param metaOutputFile
	 *            the name of the output .meta file.
	 * @throws FileNotFoundException
	 *             if file cannot be created.
	 * @throws UnsupportedEncodingException
	 *             if encoding isn't supported.
	 */
	public static void writeMetaFile(final int x, final int numPatients, final boolean one, final String metaOutputFile)
			throws FileNotFoundException, UnsupportedEncodingException {
		PrintWriter writer = new PrintWriter(metaOutputFile, "UTF-8");
		writer.println(1);
		writer.println(System.currentTimeMillis());
		writer.println(numPatients);
		if (one) {
			writer.println(1);
			writer.println(x);
		} else {
			int res = Double.valueOf(Math.ceil(Math.sqrt(x))).intValue();
			writer.println(res);
			writer.println(res);
		}
		writer.println(outputNum);
		writer.println(TRAINING_SET);
		writer.println(BATCH_SIZE);
		writer.close();

	}

	/**
	 * Based on the matrix and the pixel at hand, return the average value of
	 * the cluster.
	 * 
	 * @param matrix
	 *            containing all the gene activation data
	 * @param c
	 *            Cluster representing the pixel we want to fill
	 * @param patient
	 *            which patient we wish to pick.
	 * @return the average activation value of the cluster
	 */
	public static Double avgActivation(final List<ArrayList<Double>> matrix, final Cluster c, final int patient) {
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
	public static List<Cluster> createList(final Cluster cluster) {
		ArrayList<Cluster> list = new ArrayList<Cluster>();

		if (!cluster.getList().isEmpty()) {
			for (Cluster c : cluster.getList()) {
				list.addAll(createList(c));

			}
			return list;
		} else {
			list.add(cluster);
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
	public static Cluster[] createClusters(final Cluster[] layer1) {
		// Run nearest neighbor algorithm.
		List<Neighbors> neighbors = NearestNeighbors.neighbour(layer1);
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
	 * Helper function which finds the cluster in an list of Neighbors.class.
	 * 
	 * @param list
	 *            list of neighbors where we would like to find the cluster c.
	 * @param c
	 *            the cluster we would like to find.
	 * @return the index of cluster c.
	 */
	public static int getIndexFromNeighbors(final List<Neighbors> list, final Cluster c) {
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
	public static Cluster[] read(final InputStream in) {
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
	 * Reads the gene activation data and returns a matrix represented as a 2
	 * dimensional list.
	 * 
	 * @param in
	 *            inputstream of the file.
	 * @param numGenes
	 *            the number of elements each row.
	 * @return matrix representation of the gene activation.
	 */
	public static List<ArrayList<Double>> readGeneAct(final InputStream in, final int numGenes) {
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


}
