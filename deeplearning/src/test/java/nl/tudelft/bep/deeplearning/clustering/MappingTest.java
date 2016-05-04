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
		ArrayList<ArrayList<Integer>> result = Mapping.readGeneAct(new FileInputStream(new ClassPathResource("patients.in").getFile()),4);
		
		ArrayList<Integer> patient1 = new ArrayList<Integer>();
		patient1.add(1);
		patient1.add(2);
		patient1.add(3);
		patient1.add(4);
		ArrayList<Integer> patient2 = new ArrayList<Integer>();
		patient2.add(5);
		patient2.add(6);
		patient2.add(7);
		patient2.add(8);
		ArrayList<Integer> patient3 = new ArrayList<Integer>();
		patient3.add(9);
		patient3.add(10);
		patient3.add(11);
		patient3.add(12);
		
		ArrayList<ArrayList<Integer>> expect = new ArrayList<ArrayList<Integer>>();
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
	
}
