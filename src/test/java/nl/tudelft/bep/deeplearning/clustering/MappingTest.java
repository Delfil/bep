package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

public class MappingTest {

	public Cluster distanceRoot() {
		Cluster point1 = new Cluster(1, 2, 0);
		Cluster point2 = new Cluster(4, 5.5, 1);
		Cluster point3 = new Cluster(1, 3, 2);
		Cluster point4 = new Cluster(4, 6, 3);

		Cluster layer1_1 = new Cluster(0);
		Cluster layer1_2 = new Cluster(1);

		layer1_1.addCluster(point2);
		layer1_1.addCluster(point4);
		layer1_2.addCluster(point1);
		layer1_2.addCluster(point3);

		Cluster root = new Cluster(0);
		root.addCluster(layer1_1);
		root.addCluster(layer1_2);
		return root;
	}

	@Test
	public void testRead() throws FileNotFoundException, IOException {
		Cluster[] result = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster point1 = new Cluster(1, 2, 0);
		Cluster point2 = new Cluster(4, 5, 1);
		Cluster point3 = new Cluster(1, 3, 2);
		Cluster point4 = new Cluster(4, 6, 3);

		assertEquals(point1, result[0]);
		assertEquals(point2, result[1]);
		assertEquals(point3, result[2]);
		assertEquals(point4, result[3]);
	}

	@Test
	public void testReadGeneAct() throws FileNotFoundException, IOException {
		List<ArrayList<Double>> result = Mapping
				.readGeneAct(new FileInputStream(new ClassPathResource("test_patient.in").getFile()), 4);

		ArrayList<Double> patient1 = new ArrayList<Double>();
		patient1.add(1.0);
		patient1.add(2.0);
		patient1.add(3.0);
		patient1.add(4.0);
		ArrayList<Double> patient2 = new ArrayList<Double>();
		patient2.add(5.0);
		patient2.add(6.0);
		patient2.add(7.0);
		patient2.add(8.0);
		ArrayList<Double> patient3 = new ArrayList<Double>();
		patient3.add(9.0);
		patient3.add(10.0);
		patient3.add(11.0);
		patient3.add(12.0);

		ArrayList<ArrayList<Double>> expect = new ArrayList<ArrayList<Double>>();
		expect.add(patient1);
		expect.add(patient2);
		expect.add(patient3);

		assertEquals(expect, result);

	}

	@Test
	public void testCreateCluster() throws FileNotFoundException, IOException {
		Cluster[] input = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster[] result = Mapping.createClusters(input);

		Cluster point1 = new Cluster(1, 2, 0);
		Cluster point2 = new Cluster(4, 5, 1);
		Cluster point3 = new Cluster(1, 3, 2);
		Cluster point4 = new Cluster(4, 6, 3);

		Cluster layer1_1 = new Cluster(1, 2.5, 0);
		Cluster layer1_2 = new Cluster(4, 5.5, 1);

		layer1_1.addCluster(point1);
		layer1_1.addCluster(point3);
		layer1_2.addCluster(point2);
		layer1_2.addCluster(point4);

		Cluster[] expect = new Cluster[2];
		expect[0] = layer1_1;
		expect[1] = layer1_2;

		assertArrayEquals(expect, result);

	}

	@Test
	public void testCreateList() {
		Cluster cluster = new Cluster(0);
		Cluster deep = new Cluster(2, 3, 0);
		deep.addCluster(new Cluster(4, 5, 87));
		deep.addCluster(new Cluster(6, 5, 88));
		cluster.addCluster(deep);
		Cluster deep2 = new Cluster(3, 4, 1);
		deep2.addCluster(new Cluster(6, 7, 34));
		deep2.addCluster(new Cluster(8, 9, 35));
		cluster.addCluster(deep2);
		List<Cluster> result = Mapping.createList(cluster);

		ArrayList<Cluster> expect = new ArrayList<Cluster>();
		expect.add(new Cluster(4, 5, 87));
		expect.add(new Cluster(6, 5, 88));
		expect.add(new Cluster(6, 7, 34));
		expect.add(new Cluster(8, 9, 35));

		assertEquals(expect, result);
	}

	@Test
	public void testSorting() throws FileNotFoundException, IOException {
		Cluster[] input = Mapping.read(new FileInputStream(new ClassPathResource("pointsdistances.in").getFile()));
		Cluster[] result = Mapping.createClusters(input);
		result = Mapping.createClusters(result);

		Cluster root = distanceRoot();

		assertEquals(root, result[0]);

	}

	@Test
	public void testAvgActivation() throws FileNotFoundException, IOException {
		List<ArrayList<Double>> matrix = Mapping
				.readGeneAct(new FileInputStream(new ClassPathResource("test_patient.in").getFile()), 4);
		Cluster[] input = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster[] result = Mapping.createClusters(input);
		result = Mapping.createClusters(result);

		Double expect1_1 = 2.0;
		assertEquals(expect1_1, Mapping.avgActivation(matrix, result[0].getList().get(0), 0));
		Double expect1_2 = 3.0;
		assertEquals(expect1_2, Mapping.avgActivation(matrix, result[0].getList().get(1), 0));
		Double expect2_1 = 6.0;
		assertEquals(expect2_1, Mapping.avgActivation(matrix, result[0].getList().get(0), 1));
		Double expect3_2 = 11.0;
		assertEquals(expect3_2, Mapping.avgActivation(matrix, result[0].getList().get(1), 2));
		Double expect_root_1 = 2.5;
		assertEquals(expect_root_1, Mapping.avgActivation(matrix, result[0], 0));
		Double expect_root_2 = 6.5;
		assertEquals(expect_root_2, Mapping.avgActivation(matrix, result[0], 1));
	}

	@Test
	public void testOverallMapping() throws FileNotFoundException, IOException {
		List<ArrayList<Double>> matrix = Mapping
				.readGeneAct(new FileInputStream(new ClassPathResource("test_patient.in").getFile()), 4);
		Cluster[] points = Mapping.read(new FileInputStream(new ClassPathResource("pointsdistances.in").getFile()));
		List<Cluster> listLayer = Mapping.map(points, matrix, 1);

		List<Cluster> leafs = Mapping.map(points, matrix, 4);

		Cluster point1 = new Cluster(1, 2, 0);
		Cluster point2 = new Cluster(4, 5.5, 1);
		Cluster point3 = new Cluster(1, 3, 2);
		Cluster point4 = new Cluster(4, 6, 3);

		assertEquals(4, leafs.size());
		assertEquals(point2, leafs.get(0));
		assertEquals(point4, leafs.get(1));
		assertEquals(point1, leafs.get(2));
		assertEquals(point3, leafs.get(3));

		Cluster root = this.distanceRoot();

		assertEquals(1, listLayer.size());
		assertEquals(root, listLayer.get(0));
	}

	@Test
	public void testMain1D() throws IOException {
		String[] args = new String[5];
		args[0] = "pointsdistances.in";
		args[1] = "test_patient.in";
		args[2] = "testMainMapping";
		args[3] = "4";
		args[4] = "false";
		Mapping.main(args);

		File metaFile = new File(args[2] + ".meta");
		Scanner scanner = new Scanner(metaFile);
		scanner.nextInt();
		scanner.next();
		assertEquals(3, scanner.nextInt());
		assertEquals(2, scanner.nextInt());
		assertEquals(2, scanner.nextInt());
		assertEquals(4, scanner.nextInt());
		scanner.close();
	}

	@Test
	public void testMainNot1D() throws IOException {
		String[] args = new String[5];
		args[0] = "pointsdistances.in";
		args[1] = "test_patient.in";
		args[2] = "testMainMapping";
		args[3] = "4";
		args[4] = "true";
		Mapping.main(args);

		File dataFile = new File(args[2] + ".dat");
		Scanner scanner = new Scanner(dataFile);
		scanner.useDelimiter(",|\n");

		Double[] expected = new Double[] { 2.0, 4.0, 1.0, 3.0, 6.0, 8.0, 5.0, 7.0, 10.0, 12.0, 9.0, 11.0 };
		for (int i = 0; i < expected.length; i++) {
			Double test = scanner.nextDouble();
			assertEquals(expected[i], test, 0.001);
		}
		scanner.close();

		File metaFile = new File(args[2] + ".meta");
		scanner = new Scanner(metaFile);
		scanner.nextInt();
		scanner.next();
		assertEquals(3, scanner.nextInt());
		assertEquals(1, scanner.nextInt());
		assertEquals(4, scanner.nextInt());
		assertEquals(4, scanner.nextInt());
		scanner.close();
	}

	@Test
	public void testToLargeMin() throws FileNotFoundException, IOException {
		List<ArrayList<Double>> matrix = Mapping
				.readGeneAct(new FileInputStream(new ClassPathResource("test_patient.in").getFile()), 4);
		Cluster[] points = Mapping.read(new FileInputStream(new ClassPathResource("pointsdistances.in").getFile()));
		List<Cluster> listLayer = Mapping.map(points, matrix, 100);

		Cluster point1 = new Cluster(1, 2, 0);
		Cluster point2 = new Cluster(4, 5.5, 1);
		Cluster point3 = new Cluster(1, 3, 2);
		Cluster point4 = new Cluster(4, 6, 3);
		List<Cluster> expected = new ArrayList<Cluster>();
		expected.add(point2);
		expected.add(point4);
		expected.add(point1);
		expected.add(point3);

		assertEquals(4, listLayer.size());
		assertEquals(expected, listLayer);
	}

}
