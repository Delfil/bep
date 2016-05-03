package nl.tudelft.bep.deeplearning;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

import org.nd4j.linalg.io.ClassPathResource;

public class Mapping {

	public static void main(String[] args) throws IOException {

		Cluster[] layer1 = read(new FileInputStream(new ClassPathResource("decimal-points.in").getFile()));
		Cluster[] layer2 = new Cluster[layer1.length];
		layer2 = createClusters(layer1);

		while(layer2.length != 1) {
			layer2 = createClusters(layer2);
		}
		System.out.println(layer2[0].toString());
	}	
		
	public static Cluster[] createClusters(Cluster[] layer1) {
		int[] result1 = neighbour(layer1);
		ArrayList<Boolean> inCluster = new ArrayList<Boolean>(layer1.length);
		for(int i = 0; i < layer1.length; i++) {
			inCluster.add(i, false);
		}
		ArrayList<Cluster> layer2 = new ArrayList<Cluster>();
		int index = 0;
		int id = 0;
		Cluster cluster = new Cluster(id);
		while(inCluster.contains(false)) {
			if(!inCluster.get(index)) {
				cluster.addCluster(layer1[index]);
				inCluster.set(index, true);
				int neighbor = result1[index];
				for(int i = 0; i < layer1.length; i++) {
					if(layer1[i].getID() == neighbor) {
						index = i;
						break;
					}
				}
			}
			else {
				cluster.calculateMean();
				layer2.add(cluster);
				id++;
				cluster = new Cluster(id);
				index = inCluster.indexOf(false);		
			}
		}
		Cluster[] res = new Cluster[layer2.size()];
		int i = 0;
		for(Cluster c : layer2) {
			res[i] = c;
			i++;
		}
		return res;
	}	

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

	public static int[] neighbour(Cluster[] clusters) {
		int[] result = new int[clusters.length];
		;
		double[] distances = new double[clusters.length];
		for (int i = 0; i < clusters.length; i++) {
			distances[i] = Double.MAX_VALUE;
		}
		closestClusters(clusters, result, distances);
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
