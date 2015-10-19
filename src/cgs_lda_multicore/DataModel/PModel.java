/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.DataModel;

import cgs_lda_multicore.Utility.PLDACmdOption;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import jgibblda.Conversion;
import jgibblda.LDACmdOption;
import jgibblda.LDADataset;
import jgibblda.Model;

/**
 * Prepare for GPU parallel LDA algorithm. 
 * Reuse old Document and LDADataset. Partition the DW (Z), DT, WT array into 
 * many small array based on partition of D and W. 
 * T is copy into P array for P core.
 * 
 * Then, each core access its small array of DW, DT, and WT in turn (in estimate/inference method).
 * AND NOTICE THAT: the way to partition data is still a room for improvement.
 * 
 * Note: with assumption that all docs and words are consecutively indexed 
 * from 0 to M - 1 and from 0 to V - 1 respectively.
 * That means index can contain even word that is not present in any documents.
 * 
 * @author THNghiep
 */
public class PModel extends jgibblda.Model {
    // General use variable.
    public int numCumulativeSample;
    public float[][] stationaryTheta;
    public float[][] stationaryPhi;
    String stationaryThetaSuffix;
    String stationaryPhiSuffix;
    
    // General use option.
    public int burnIn;
    public int howToComputeDistribution;
    public double testSetProportion;
    public String dfiletrain;
    public String dfiletest;
    
    // Use for parallel processing.
    /**
     * Maximum number of threads to run in parallel.
     */
    public int threadPoolSize;
    /**
     * Number of partition.
     */
    public int P;
    public int shuffleTimes;
    public int howToPartition;
    
    /**
     * Mapping doc and word to partition.
     * For size efficiency and avoid rehash and hash collision: init with preknown max capacity and high load factor.
     * Most important problem is rehash and hash collision, size is not critical.
     * Note: rehash happen when reallocate, hash collision happen even when the number of item < hashmap size.
     * So init size need to be large, about 2 max capacity, and load factor = max = 1.
     * 
     * Note 2: could optimize more by using int array instead of hashmap because docid and wordid is consecutive.
     * But here use hashmap for flexibility and extendability, 
     * will change if size and performance are critical.
     * Currently < x00 MB.
     */
    public HashMap<Integer, Integer> docToPartition;
    public HashMap<Integer, Integer> wordToPartition;
    
    /**
     * Mapping partition to Doc and Word.
     * For size efficiency: 1, init with preknown max capacity so it will not reallocate, 2, trimToSize().
     * Note: do not need to allocate abundantly, trim when in doubt.
     * 
     * Note 2: could optimize more by using jagged array instead of arraylist of arraylist, using array of hashset instead of arraylist of hashset.
     * But currently < x00 MB, will change if critical.
     */
    public ArrayList<ArrayList<Integer>> partitionToDoc;
    public ArrayList<HashSet<Integer>> partitionToWord;

    /**
     * List of docID and wordID with the order we use to decide partition.
     */
    public List<Integer> docIDList;
    public List<Integer> wordIDList;

    /**
     * The ending excluding index of doc and word in each partition.
     */
    public int[] docPartitionEndingSuperSimple;
    public int[] wordPartitionEndingSuperSimple;
    
    /**
     * Array of all eta.
     */
    public double[] etaArray;
    /**
     * Efficient partition quotient.
     */
    public double etaBest;
    /**
     * Average eta of all many shuffle.
     */
    public double etaRandom;
    
    /**
     * Replicate of T matrix.
     */
//    public List<int[]> nwsumList;
    public int[][] nwsumList;
    /**
     * Replicate of p.
     * Reused temp variable, to avoid declaring new variable in each word instance sampling.
     */
    public double[][] pList;
    
    public PModel() {
        super();
        // General use.
        this.numCumulativeSample = 0;
        this.stationaryTheta = null;
        this.stationaryPhi = null;
        this.stationaryThetaSuffix = ".statheta";
        this.stationaryPhiSuffix = ".staphi";
    }

    /**
     * This method make partition for data.
     * Data to partition: 
     * DW: z
     * DT: nd
     * WT: nw
     * Data to replicate:
     * T: nwsum
     * First, partition: 
     * D: data.docs
     * W: data.localDict.id2word.keySet()
     * 
     * The original order of Doc in data.docs 
     * and the original order of Word in data.localDict.id2word are reserved.
     * 
     * Partition result will be hash map that show: doc -> partition and word -> partition.
     * And mapping from partition to doc and word.
     * Also have shuffle order of doclist and wordlist that is used to create partition.
     * 
     * Then, if needed, it can be used to rearrange docs[] & dictionary to use with doc/word ending index.
     * And can even build separate partition variables.
     * 
     * @param howToPartition: 1: even partition, 2: gpu partition.
     * @throws Exception 
     */
    public void partitionData(int howToPartition) throws Exception {
        // Get newly created list.
        getDocIDList();
        getWordIDList();
        
        // New list contain same Integer object.
        // However it could add/remove/change order independently, so shuffle is OK.
        List<Integer> docIDListBest = new ArrayList<>(docIDList);
        List<Integer> wordIDListBest = new ArrayList<>(wordIDList);
        
        // Partition method will newly create a HashMap, so old content is reserved.
        HashMap<Integer, Integer> docToPartitionBest = docToPartition;
        HashMap<Integer, Integer> wordToPartitionBest = wordToPartition;
        
        etaArray = new double[shuffleTimes];
        etaBest = 0;
        etaRandom = 0;
        
        System.out.println("Partition into " + P + " and compute Eta. Parallel in " + threadPoolSize + " threads.");
        for (int i = 0; i < shuffleTimes; i++) {
            long start = System.currentTimeMillis();
            
            if (howToPartition == 1) {
                evenPartition();
            } else if (howToPartition == 2) {
                gpuPartitionParallel();
            }
            
            etaArray[i] = computeEtaParallel();
            if (etaArray[i] > etaBest) {
                docIDListBest = new ArrayList<>(docIDList);
                wordIDListBest = new ArrayList<>(wordIDList);
                docToPartitionBest = docToPartition;
                wordToPartitionBest = wordToPartition;
                etaBest = etaArray[i];
            }

            // Shuffle after partitioning, so the first partitioning is based on original order.
            Collections.shuffle(docIDList);
            Collections.shuffle(wordIDList);

            long elapse = System.currentTimeMillis() - start;
            System.out.println("Shuffling " + (i + 1) + ". Eta = " + etaArray[i] + ". In: " + (double) elapse / 1000 + " second.");
        }
        docIDList = docIDListBest;
        wordIDList = wordIDListBest;
        docToPartition = docToPartitionBest;
        wordToPartition = wordToPartitionBest;
        partitionToDoc();
        partitionToWord();
        
        for (double eta : etaArray) {
            etaRandom += eta;
        }
        etaRandom /= shuffleTimes;
        
        // Replicate data.
        replicateTWSumMatrix();
        replicateTempTopicProb();
    }
    
    /**
     * Simple generate list from 0 to M-1. With assumption all docs are indexed from 0 to M - 1.
     * @throws Exception 
     */
    private void getDocIDList() throws Exception {
        docIDList = new ArrayList<>(M);
        for (int i = 0; i < M; i++) {
            docIDList.add(i);
        }
    }
    
    /**
     * Simple generate list from 0 to V-1. With assumption all words in vocabulary are indexed from 0 to V - 1.
     * @throws Exception 
     */
    private void getWordIDList() throws Exception {
        wordIDList = new ArrayList<>(V);
        for (int i = 0; i < V; i++) {
            wordIDList.add(i);
        }
    }

    /**
     * Super simple algorithm, divide by number of docs and words. 
     * Doc and word is indexed from 0, consecutive.
     * Result are doc and word partition ending.
     * 
     * @throws Exception 
     */
    private void superSimplePartition() throws Exception {
        docPartitionEndingSuperSimple = new int[P];
        wordPartitionEndingSuperSimple = new int[P];
        
        for (int i = 0; i < P - 1; i++) {
            docPartitionEndingSuperSimple[i] = M / P * (i + 1);
            wordPartitionEndingSuperSimple[i] = V / P * (i + 1);
        }
        docPartitionEndingSuperSimple[P - 1] = M;
        wordPartitionEndingSuperSimple[P - 1] = V;
    }
    
    /**
     * Simple algorithm, divide by number of docs and words. 
     * Doc and word are not consecutive. 
     * Result are doc and word partition hash map. 
     * Resulted HashMap is newly created.
     * 
     * @throws Exception 
     */
    private void evenPartition() throws Exception {
        docToPartition = new HashMap<>(2 * M, 1f);
        wordToPartition = new HashMap<>(2 * V, 1f);
        
        for (int i = 0; i < P; i++) {
            // Get range of doc and word for each partition.
            int docStart = i * M / P;
            int docEnd = (i + 1) * M / P;
            if (docEnd > M) {
                docEnd = M;
            }
            int wordStart = i * V / P;
            int wordEnd = (i + 1) * V / P;
            if (wordEnd > V) {
                wordEnd = V;
            }
            
            // Put word and doc in the specified range to partition map.
            for (int j = docStart; j < docEnd; j++) {
                docToPartition.put(docIDList.get(j), i);
            }
            for (int j = wordStart; j < wordEnd; j++) {
                wordToPartition.put(wordIDList.get(j), i);
            }
        }
    }

    /**
     * Algorithm in GPU paper, divide by number of word instances in rows and columns. 
     * Doc and word are not consecutive. 
     * Result are doc and word partition hash map. 
     * Resulted HashMap is newly created.
     * 
     * @throws Exception 
     */
    private void gpuPartitionParallel() throws Exception {
        docToPartition = new HashMap<>(2 * M, 1f);
        wordToPartition = new HashMap<>(2 * V, 1f);

        // Total number Word Instance.
        final AtomicLong S = new AtomicLong();
        // Number of Word Instance in each doc and of each word.
        final AtomicLong[] docWordInstance = new AtomicLong[M];
        final AtomicLong[] wordWordInstance = new AtomicLong[V];
        for (int i = 0; i < M; i++) {
            docWordInstance[i] = new AtomicLong();
        }
        for (int i = 0; i < V; i++) {
            wordWordInstance[i] = new AtomicLong();
        }
        
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);
        for (int m = 0; m < M; m++) {
            final int docID = m;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        // Count word instances.
                        S.addAndGet(data.docs[docID].length);
                        docWordInstance[docID].set(data.docs[docID].length);
                        for (Integer wordID : data.docs[docID].words) {
                            wordWordInstance[wordID].incrementAndGet();
                        }
                    } catch (Exception ex) {
                        System.err.println(ex.toString());
                        ex.printStackTrace();
                    }
                }
            });
        }
        // Shutdown thread pool, wait until shutdown finished.
        executor.shutdown();
        while (!executor.isTerminated()) {}
        
        // Compute optimal C.
        double Copt = S.doubleValue() / P;

        // Begin partitioning.
        int part = 0;
        long sumWordInstance = 0;
        double goal = Copt;
        double gap = Copt;
        double oldGap;
        for (int i = 0; i < M; i++) {
            // This for loop divides documents into P even groups, 
            // with regard to number of word instances in each document.
            int docID = docIDList.get(i);
            sumWordInstance += docWordInstance[docID].get();
            
            // Keep info about old gap.
            oldGap = gap;
            // Compute new gap for new document.
            gap = Math.abs(goal - sumWordInstance);
            if (gap > oldGap) {
                // Move to new partition.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S.get();
                }
                gap = Math.abs(goal - sumWordInstance);
            }
            
            docToPartition.put(docID, part);
        }

        part = 0;
        sumWordInstance = 0;
        goal = Copt;
        gap = Copt;
        for (int i = 0; i < V; i++) {
            // This for loop divides words (vocabulary) into P even groups, 
            // with regard to number of word instances of each word.
            int wordID = wordIDList.get(i);
            sumWordInstance += wordWordInstance[wordID].get();
            
            // Keep info about old gap.
            oldGap = gap;
            // Compute new gap for new document.
            gap = Math.abs(goal - sumWordInstance);
            if (gap > oldGap) {
                // Move to new partition.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S.get();
                }
                gap = Math.abs(goal - sumWordInstance);
            }
            
            wordToPartition.put(wordID, part);
        }
    }

    /**
     * Algorithm in GPU paper, divide by number of word instances in rows and columns. 
     * Doc and word are not consecutive. 
     * Result are doc and word partition hash map. 
     * Resulted HashMap is newly created.
     * 
     * @throws Exception 
     */
    private void gpuPartition() throws Exception {
        docToPartition = new HashMap<>(2 * M, 1f);
        wordToPartition = new HashMap<>(2 * V, 1f);
        
        int[] docWordInstance = new int[M];
        int[] wordWordInstance = new int[V];

        // Total number Word Instance.
        long S = 0;

        // Compute optimal C.
        for (int docID = 0; docID < M; docID++) {
            // Count word instances.
            S += data.docs[docID].length;
            docWordInstance[docID] = data.docs[docID].length;
            for (Integer wordID : data.docs[docID].words) {
                wordWordInstance[wordID] += 1;
            }
        }
        double Copt = (double) S / P;

        // Begin partitioning.
        int part = 0;
        long sumWordInstance = 0;
        double goal = Copt;
        double gap = Copt;
        double oldGap;
        for (int i = 0; i < M; i++) {
            // This for loop divides documents into P even groups, 
            // with regard to number of word instances in each document.
            int docID = docIDList.get(i);
            sumWordInstance += docWordInstance[docID];
            
            // Keep info about old gap.
            oldGap = gap;
            // Compute new gap for new document.
            gap = Math.abs(goal - sumWordInstance);
            if (gap > oldGap) {
                // Move to new partition.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S;
                }
                gap = Math.abs(goal - sumWordInstance);
            }
            
            docToPartition.put(docID, part);
        }

        part = 0;
        sumWordInstance = 0;
        goal = Copt;
        gap = Copt;
        for (int i = 0; i < V; i++) {
            // This for loop divides words (vocabulary) into P even groups, 
            // with regard to number of word instances of each word.
            int wordID = wordIDList.get(i);
            sumWordInstance += wordWordInstance[wordID];
            
            // Keep info about old gap.
            oldGap = gap;
            // Compute new gap for new document.
            gap = Math.abs(goal - sumWordInstance);
            if (gap > oldGap) {
                // Move to new partition.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S;
                }
                gap = Math.abs(goal - sumWordInstance);
            }
            
            wordToPartition.put(wordID, part);
        }
    }

    /**
     * Count in parallel to compute Eta.
     * 
     * @return Eta.
     * @throws Exception 
     */
    private double computeEtaParallel() throws Exception {
        // Total number Word Instance.
        final AtomicLong S = new AtomicLong();
        // Number of Word Instance in each Partition in DW, row is doc, column is word.
        final AtomicLong[][] Cmn = new AtomicLong[P][P];
        for (int i = 0; i < P; i++) {
            for (int j = 0; j < P; j++) {
                Cmn[i][j] = new AtomicLong();
            }
        }
        
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);
        for (int m = 0; m < M; m++) {
            final int docID = m;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        // Compute optimal C.
                        S.addAndGet(data.docs[docID].length);
                        for (Integer wordID : data.docs[docID].words) {
                            Cmn[docToPartition.get(docID)][wordToPartition.get(wordID)].incrementAndGet();
                        }
                    } catch (Exception ex) {
                        System.err.println(ex.toString());
                        ex.printStackTrace();
                    }
                }
            });
        }
        // Shutdown thread pool, wait until shutdown finished.
        executor.shutdown();
        while (!executor.isTerminated()) {}
        
        // Compute optimal C.
        double Copt = S.doubleValue() / P;
        
        // Compute real C. Using computed Cmn matrix, which is small, get max of each diagonal line, then sum to get C.
        long[] maxCmn = new long[P];
        for (int col = 0; col < P; col++) {
            for (int row = 0; row < P; row++) {
                if (maxCmn[col] < Cmn[row][(row + col) % P].get()) {
                    maxCmn[col] = Cmn[row][(row + col) % P].get();
                }
            }
        }
        double C = 0;
        for (long max : maxCmn) {
            C += max;
        }
        
        // Compute eta.
        double eta = Copt / C;
        return eta;
    }

    /**
     * Loop over data.
     * 
     * @return eta: efficient partition quotient.
     * @throws Exception 
     */
    private double computeEta() throws Exception {
        // Total number Word Instance.
        long S = 0;
        // Number of Word Instance in each Partition in DW, row is doc, column is word.
        long[][] Cmn = new long[P][P];
        
        // Compute optimal C.
        for (int docID = 0; docID < M; docID++) {
            S += data.docs[docID].length;
            for (Integer wordID : data.docs[docID].words) {
                Cmn[docToPartition.get(docID)][wordToPartition.get(wordID)]++;
            }
        }
        double Copt = (double) S / P;
        
        // Compute real C.
        long[] maxCmn = new long[P];
        for (int col = 0; col < P; col++) {
            for (int row = 0; row < P; row++) {
                if (maxCmn[col] < Cmn[row][(row + col) % P]) {
                    maxCmn[col] = Cmn[row][(row + col) % P];
                }
            }
        }
        double C = 0;
        for (long max : maxCmn) {
            C += max;
        }
        
        // Compute eta.
        double eta = Copt / C;
        return eta;
    }
    
    /**
     * Sync replicated data.
     * @throws Exception 
     */
    public void syncTWSum() throws Exception {
        computeTWSumMatrix();
        replicateTWSumMatrix();
    }

    /**
     * Recompute T matrix based on all replication of it.
     * @throws Exception 
     */
    private void computeTWSumMatrix() throws Exception {
        int[] newT = new int[K];
        
        for (int k = 0; k < K; k++) {
            for (int p = 0; p < P; p++) {
//                newT[k] += nwsumList.get(p)[k] - nwsum[k];
                newT[k] += nwsumList[p][k] - nwsum[k];
            }
            newT[k] += nwsum[k];
        }
        
        nwsum = newT;
    }
    
    /**
     * Make replication of T matrix.
     * @throws Exception 
     */
    private void replicateTWSumMatrix() throws Exception {
//        nwsumList = new ArrayList<>(P);
        nwsumList = new int[P][];
        for (int i = 0; i < P; i++) {
            // .clone() or Arrays.copyOf() will deep copy to 1 level.
            // That is, it works for 1-D array, not 2-D array or ArrayList of object.
//            nwsumList.add(Arrays.copyOf(nwsum, K));
            nwsumList[i] = Arrays.copyOf(nwsum, K);
        }
    }
    
    /**
     * Make replication of Temp Topic Prob.
     * @throws Exception 
     */
    private void replicateTempTopicProb() throws Exception {
        pList = new double[P][];
        for (int i = 0; i < P; i++) {
            pList[i] = Arrays.copyOf(p, K);
        }
    }
    
    /**
     * Mapping partition to doclist.
     * @throws Exception 
     */
    private void partitionToDoc() throws Exception {
        partitionToDoc = new ArrayList<>(P);
        for (int i = 0; i < P; i++) {
            partitionToDoc.add(new ArrayList<Integer>());
        }
        for (int docID : docToPartition.keySet()) {
            partitionToDoc.get(docToPartition.get(docID)).add(docID);
        }
        for (int i = 0; i < P; i++) {
            partitionToDoc.get(i).trimToSize();
        }
    }
    
    /**
     * Mapping partition to wordset.
     * @throws Exception 
     */
    private void partitionToWord() throws Exception {
        partitionToWord = new ArrayList<>();
        for (int i = 0; i < P; i++) {
            partitionToWord.add(new HashSet<Integer>());
        }
        for (int wordID : wordToPartition.keySet()) {
            partitionToWord.get(wordToPartition.get(wordID)).add(wordID);
        }
    }

    /**
     * Theta is computed in the same way for train or new model.
     */
    public void computeTheta() {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                theta[m][k] = (float) ((nd[m][k] + alpha) / (ndsum[m] + K * alpha));
            }
        }
    }

    /**
     * Phi is computed in different ways for train and new model.
     */
    public void computePhi() {
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                phi[k][w] = (float) ((nw[w][k] + beta) / (nwsum[k] + V * beta));
            }
        }
    }

    /**
     * Theta is computed in the same way for train or new model.
     * Call to computeTheta().
     */
    public void computeNewTheta() {
        computeTheta();
    }

    /**
     * THNghiep: Old code seems to have error: repeated newModel.nwsum[k] in denominator.
     * Here change to trnModel.nwsum[].
     * 
     * @param trnModel estimated model used as prior to compute new Phi.
     */
    public void computeNewPhi(Model trnModel) {
        for (int k = 0; k < K; k++) {
            for (int _w = 0; _w < V; _w++) {
                Integer id = data.lid2gid.get(_w);
                if (id != null) {
                    phi[k][_w] = (float) ((trnModel.nw[id][k] + nw[_w][k] + beta) 
                            / (trnModel.nwsum[k] + nwsum[k] + trnModel.V * beta));
                }
            }//end foreach word
        }// end foreach topic
    }


    public void computeCumulativeTheta() {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                stationaryTheta[m][k] += theta[m][k];
            }
        }
    }

    public void computeCumulativePhi() {
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                stationaryPhi[k][w] += phi[k][w];
            }
        }
    }

    public void computeStationaryTheta() {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                stationaryTheta[m][k] /= numCumulativeSample;
            }
        }
    }

    public void computeStationaryPhi() {
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                stationaryPhi[k][w] /= numCumulativeSample;
            }
        }
    }
    
    /**
     * Save model
     */
    @Override
    public boolean saveModel(String modelName) {
        super.saveModel(modelName);

        if (!saveModelStationaryTheta(dir + File.separator + modelName + stationaryThetaSuffix)) {
            return false;
        }

        if (!saveModelStationaryPhi(dir + File.separator + modelName + stationaryPhiSuffix)) {
            return false;
        }
        
        return true;
    }

    /**
     * Save theta (topic distribution) for this model
     */
    public boolean saveModelStationaryTheta(String filename) {
        if (stationaryTheta == null) {
            return true;
        }
        
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < K; j++) {
                    writer.write(stationaryTheta[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving topic distribution file for this model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Save word-topic distribution
     */
    public boolean saveModelStationaryPhi(String filename) {
        if (stationaryPhi == null) {
            return true;
        }
        
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

            for (int i = 0; i < K; i++) {
                for (int j = 0; j < V; j++) {
                    writer.write(stationaryPhi[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving word-topic distribution:" + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * initialize the model
     */
    protected boolean init(PLDACmdOption option) {
        if (option == null) {
            return false;
        }

        super.init(option);

        // General use.
        dfiletrain=  option.dfiletrain;
        dfiletest=  option.dfiletest;
        burnIn = option.burnIn;
        howToComputeDistribution = option.howToGetDistribution;
        testSetProportion = option.testSetProportion;

        // Use for parallel processing.
        if (option.threadPoolSize > 0) {
            this.threadPoolSize = option.threadPoolSize;
        } else {
            this.threadPoolSize = option.P;
        }
        this.P = option.P;
        this.shuffleTimes = option.shuffleTimes;
        howToPartition = option.howToPartition;

        return true;
    }

    /**
     * Init parameters for estimation. Prepare training set and test set from shared dataset.
     * 
     * @param option
     * @return 
     * @throws java.lang.Exception
     */
    public boolean initNewModel(PLDACmdOption option) throws Exception {
        if (!init(option)) {
            return false;
        }

        int m, n, w, k;
        p = new double[K];

        readData(option);

        // allocate memory and assign values for variables		
        M = data.M;
        V = data.V;
        dir = option.dir;
        savestep = option.savestep;

        // K: from command line or default value
        // alpha, beta: from command line or default values
        // niters, savestep: from command line or default values
        nw = new int[V][K];
        for (w = 0; w < V; w++) {
            for (k = 0; k < K; k++) {
                nw[w][k] = 0;
            }
        }

//        nd = new int[M][K];
        nd = new short[M][K];
        for (m = 0; m < M; m++) {
            for (k = 0; k < K; k++) {
                nd[m][k] = 0;
            }
        }

        nwsum = new int[K];
        for (k = 0; k < K; k++) {
            nwsum[k] = 0;
        }

        ndsum = new int[M];
        for (m = 0; m < M; m++) {
            ndsum[m] = 0;
        }

//        z = new Vector[M];
//        z = new ArrayList[M];
        z = new short[M][];
        for (m = 0; m < data.M; m++) {
            int N = data.docs[m].length;
//            z[m] = new Vector<Integer>();
//            z[m] = new ArrayList<Integer>();
            z[m] = new short[N];

            //initilize for z
            for (n = 0; n < N; n++) {
                int topic = (int) Math.floor(Math.random() * K);
//                z[m].add(topic);
                z[m][n] = (short) topic;

                // number of instances of word assigned to topic j
                nw[data.docs[m].words[n]][topic] += 1;
                // number of words in document i assigned to topic j
                nd[m][topic] += 1;
                // total number of words assigned to topic j
                nwsum[topic] += 1;
            }
            // total number of words in document i
            ndsum[m] = N;
        }

//        theta = new double[M][K];
//        phi = new double[K][V];
        theta = new float[M][K];
        phi = new float[K][V];
        
        if (option.howToGetDistribution == 2) {
            stationaryTheta = new float[M][K];
            stationaryPhi = new float[K][V];
        }

        return true;
    }

    /**
     * Read data from multiple formated datafile datasest to LDADataset object, separate train and test set.
     * 
     * @param option - if there are only dfile then read and separate test set, 
     * if there are both dfiletrain and dfiletest then read train and test set corresponding to word map file.
     * @return true if read successfully.
     * @throws Exception 
     */
    public boolean readData(PLDACmdOption option) throws Exception {
        if (!option.dfile.isEmpty()) {
            // First time read data.
            // Training set.
            if (option.datafileFormat.trim().equalsIgnoreCase("Private")) {
                data = LDADataset.readDataSet(dir + File.separator + dfile);
            } else if (option.datafileFormat.trim().equalsIgnoreCase("NYT")) {
                data = DataPreparation.readDatasetNYT(dir + File.separator + dfile, dir + File.separator + wordMapFile);
            } else if (option.datafileFormat.trim().equalsIgnoreCase("CORE")) {
                // Not implemented yet.
            }
            // Save wordmap right after reading from datafile.
            data.localDict.writeWordMap(option.dir + File.separator + "WordMap_" + option.dfile);
            // Separate Test set.
            if (option.isSepTestset) {
                dataTest = DataPreparation.prepareTestSet(data, option.testSetProportion, dir, dfile);
            }
        } else if (!option.dfiletrain.isEmpty() && !option.dfiletest.isEmpty()) {
            data = DataPreparation.readDataset(dir + File.separator + dfiletrain, dir + File.separator + wordMapFile);
            dataTest = DataPreparation.readDataset(dir + File.separator + dfiletest, dir + File.separator + wordMapFile);
        } else {
            return false;
        }

        return true;
    }
}