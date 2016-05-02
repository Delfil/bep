package nl.tudelft.bep.deeplearning;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

public class Mapping {

	static int[] result;
	static double[] distances;
	static Cluster[] Clusters;

	public static void main(String[] args) {

		read(null);
		neighbour();
	}

	public static void read(InputStream in) {
		Scanner scanner = new Scanner(new InputStreamReader(in));

		int n = scanner.nextInt();
		result = new int[n];
		distances = new double[n];
		for (int i = 0; i < n; i++) {
			distances[i] = Double.MAX_VALUE;
		}

		Clusters = new Cluster[n];

		for (int i = 0; i < n; i++) {
			Clusters[i] = new Cluster(scanner.nextFloat(), scanner.nextFloat(), i);
		}
		scanner.close();
	}

	public static int[] neighbour() {
		closestClusters(Clusters, result, distances);
		return result;
	}

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
