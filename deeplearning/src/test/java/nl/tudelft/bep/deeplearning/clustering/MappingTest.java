package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

public class MappingTest {

	@Test
	public void testRead() throws FileNotFoundException, IOException {
		Cluster[] result = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster point1 = new Cluster(1,2,0);
		Cluster point2 = new Cluster(4,5,1);
		Cluster point3 = new Cluster(1,3,2);
		Cluster point4 = new Cluster(4,6,3);
		
		assertEquals(point1, result[0]);
		assertEquals(point2, result[1]);
		assertEquals(point3, result[2]);
		assertEquals(point4, result[3]);
	}
	
	@Test
	public void testReadGeneAct() throws FileNotFoundException, IOException {
		ArrayList<ArrayList<Double>> result = Mapping.readGeneAct(new FileInputStream(new ClassPathResource("patients.in").getFile()),4);
		
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
		
		Cluster point1 = new Cluster(1,2,0);
		Cluster point2 = new Cluster(4,5,1);
		Cluster point3 = new Cluster(1,3,2);
		Cluster point4 = new Cluster(4,6,3);
		
		Cluster layer1_1 = new Cluster(1,2.5,0);
		Cluster layer1_2 = new Cluster(4,5.5,1);
		
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
		Cluster deep = new Cluster(2,3,0);
		deep.addCluster(new Cluster(4,5,87));
		deep.addCluster(new Cluster(6,5,88));
		cluster.addCluster(deep);
		Cluster deep2 = new Cluster(3,4,1);
		deep2.addCluster(new Cluster(6,7,34));
		deep2.addCluster(new Cluster(8,9,35));
		cluster.addCluster(deep2);
		ArrayList<Integer> result = Mapping.createList(cluster);
		
		ArrayList<Integer> expect = new ArrayList<Integer>();
		expect.add(87);
		expect.add(88);
		expect.add(34);
		expect.add(35);
		
		assertEquals(expect, result);	
	}
	
	@Test 
	public void testLayerSize() throws FileNotFoundException, IOException {
		Cluster[] input = Mapping.read(new FileInputStream(new ClassPathResource("points.in").getFile()));
		Cluster[] result = Mapping.createClusters(input);
		result = Mapping.createClusters(result);	
		
		assertEquals(2, Mapping.layerSize(result[0], 1));
		assertEquals(4, Mapping.layerSize(result[0], 2));
		
	}	
	
}
