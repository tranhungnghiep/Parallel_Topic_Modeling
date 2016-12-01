/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.DataModel;

import cgs_lda_multicore.Utility.GeneralUtility;
import cgs_lda_multicore.Utility.PLDACmdOption;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import jgibblda.Conversion;
import jgibblda.LDACmdOption;
import jgibblda.LDADataset;
import jgibblda.Model;
import jgibblda.Pair;

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
public class PModel_BoT extends PModel {
    public float[][] stationaryPi;
    String stationaryPiSuffix;
    String piSuffix;
    String tstassignSuffix;
    String ttssSuffix;
    
    // --------------------------------------
    // For BoT model.
    // --------------------------------------
    public LDADataset_BoT data;
    public LDADataset_BoT dataTest;
    /**
     * Number of unique timestamp tokens, size of timestamp set.
     */
    public int O;
    public int L;
    /**
     * BoT  hyperparameters
     */
    public double gamma; 
    
    /**
     * T-TS Proportion: pi: topic-timestamp distributions, size K x O
     */
    public float[][] pi;
    /**
     * D-TS: Topic assigned to timestamp token, size M x L.
     */
    public short[][] y;
    /**
     * TS-T: nts[i][j]: number of instances of timestamp token i assigned to topic j, size O x K
     */
    public int[][] nts;
    /**
     * T of TS, not T of word: ntssum[j]: total number of timestamp tokens assigned to topic j, size K. 
     * Not use directly in PLDA, need to replicate.
     */
    public int[] ntssum;

    // --------------------------------------
    // For parallel.
    // --------------------------------------
    /**
     * Replicate of TTS matrix.
     */
    public int[][] ntssumList;
    
    // --------------------------------------
    // For partitioning timestamp.
    // --------------------------------------
    // BoT Partition, approach 1: partition z and y separately. 
    // Then sampling z and y consecutively in one turn (either 1 diagonal, or 1 full iteration).
    public int shuffleTimesTS;
    public HashMap<Integer, Integer> docToPartitionBoTA1;
    public ArrayList<ArrayList<Integer>> partitionToDocBoTA1;
    public HashMap<Integer, Integer> tsToPartitionBoTA1;
    public ArrayList<HashSet<Integer>> partitionToTSBoTA1;
    public List<Integer> docIDListBoTA1;
    public List<Integer> tsIDListBoTA1;
    /**
     * Array of all eta for Timestamp.
     */
    public double[] etaTSArrayBoTA1;
    /**
     * Efficient partition quotient for Timestamp.
     */
    public double etaTSBestBoTA1;
    /**
     * Average eta of all many shuffle for Timestamp.
     */
    public double etaTSRandomBoTA1;    
    
    // --------------------------------------
    // For partitioning both data and timestamp.
    // --------------------------------------
    // BoT Partition, approach 2.
    
    public PModel_BoT() {
        super();
        this.stationaryPi = null;
        this.stationaryPiSuffix = ".stapi";
        this.piSuffix = ".pi";
        this.tstassignSuffix = ".tstassign";
        this.ttssSuffix = ".ttss";
    }

    /**
     * Call after partitioning Data.
     * Partitioning data method already produced docIDList and wordIDList, doc to partition and word to partition.
 Need to shuffle tsIDListBoTA1 and build ts to partition.
     * 
     * @param howToPartition
     * @throws Exception 
     */
    public void partitionTS(int howToPartition) throws Exception {
        // Get newly created list.
        getDocIDListBoTA1();
        getTSIDListBoTA1();
        
        // Permute dw matrix and get doc and word id list.
        if (this.permuteAlgorithm.startsWith("A1") || this.permuteAlgorithm.startsWith("A2")) {
            dtsPermuteParallel();
        }
        
        // New list contain same Integer object.
        // However it could add/remove/change order independently, so shuffle is OK.
        List<Integer> docIDListTSBest = new ArrayList<>(docIDListBoTA1);
        List<Integer> tsIDListBest = new ArrayList<>(tsIDListBoTA1);
        
        // Partition method will newly create a HashMap, so old content is reserved.
        HashMap<Integer, Integer> docToPartitionTSBest = docToPartitionBoTA1;
        HashMap<Integer, Integer> tsToPartitionBest = tsToPartitionBoTA1;
        
        etaTSArrayBoTA1 = new double[shuffleTimesTS];
        etaTSBestBoTA1 = 0;
        etaTSRandomBoTA1 = 0;
        
        System.out.println("Partition Timestamp into " + P + " and compute Eta. Parallel in " + threadPoolSize + " threads.");
        for (int i = 0; i < shuffleTimesTS; i++) {
            long start = System.currentTimeMillis();
            
            if (howToPartition == 1) {
                evenPartitionTS();
            } else if (howToPartition == 2) {
                gpuPartitionTSParallel();
            }
            
            etaTSArrayBoTA1[i] = computeEtaTSParallel();
            if (etaTSArrayBoTA1[i] > etaTSBestBoTA1) {
                docIDListTSBest = new ArrayList<>(docIDListBoTA1);
                tsIDListBest = new ArrayList<>(tsIDListBoTA1);
                docToPartitionTSBest = docToPartitionBoTA1;
                tsToPartitionBest = tsToPartitionBoTA1;
                etaTSBestBoTA1 = etaTSArrayBoTA1[i];
            }

            // Shuffle after partitioning, so the first partitioning is based on original order.
            if (this.permuteAlgorithm.startsWith("A2")) {
                dtsPermuteParallel();
            } else {
                Collections.shuffle(docIDListBoTA1);
                Collections.shuffle(tsIDListBoTA1);
            }

            long elapse = System.currentTimeMillis() - start;
            System.out.println("TS shuffling " + (i + 1) + ". Eta = " + etaTSArrayBoTA1[i] + ". In: " + (double) elapse / 1000 + " second.");
        }
        docIDListBoTA1 = docIDListTSBest;
        tsIDListBoTA1 = tsIDListBest;
        docToPartitionBoTA1 = docToPartitionTSBest;
        tsToPartitionBoTA1 = tsToPartitionBest;
        partitionToDocBoTA1();
        partitionToTSBoTA1();
        
        for (double eta : etaTSArrayBoTA1) {
            etaTSRandomBoTA1 += eta;
        }
        etaTSRandomBoTA1 /= shuffleTimesTS;
        
        // Replicate data.
        replicateTTSSumMatrix();
    }

    /**
     * Simple generate list from 0 to M-1. With assumption all docs are indexed from 0 to M - 1.
     * @throws Exception 
     */
    private void getDocIDListBoTA1() throws Exception {
        docIDListBoTA1 = new ArrayList<>(M);
        for (int i = 0; i < M; i++) {
            docIDListBoTA1.add(i);
        }
    }
    
    /**
     * Simply generate list of TS ID from 0 to O-1. 
     * Data input method needs to guarantee that ts id are consecutive and are indexed from 0 to O - 1.
     * @throws Exception 
     */
    private void getTSIDListBoTA1() throws Exception {
        tsIDListBoTA1 = new ArrayList<>(O);
        for (int i = 0; i < O; i++) {
            tsIDListBoTA1.add(i);
        }
    }

    /**
     * Mapping partition to doclist.
     * @throws Exception 
     */
    private void partitionToDocBoTA1() throws Exception {
        partitionToDocBoTA1 = new ArrayList<>(P);
        for (int i = 0; i < P; i++) {
            partitionToDocBoTA1.add(new ArrayList<Integer>());
        }
        for (int docID : docToPartitionBoTA1.keySet()) {
            partitionToDocBoTA1.get(docToPartitionBoTA1.get(docID)).add(docID);
        }
        for (int i = 0; i < P; i++) {
            partitionToDocBoTA1.get(i).trimToSize();
        }
    }
    
    /**
     * Mapping partition to timestamp id set.
     * @throws Exception 
     */
    private void partitionToTSBoTA1() throws Exception {
        partitionToTSBoTA1 = new ArrayList<>();
        for (int i = 0; i < P; i++) {
            partitionToTSBoTA1.add(new HashSet<Integer>());
        }
        for (int tsID : tsToPartitionBoTA1.keySet()) {
            partitionToTSBoTA1.get(tsToPartitionBoTA1.get(tsID)).add(tsID);
        }
    }

    /**
     * Count in parallel to compute Eta for TS.
     * 
     * @return Eta TS.
     * @throws Exception 
     */
    private double computeEtaTSParallel() throws Exception {
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
                        S.addAndGet(data.docs[docID].L);
                        for (Integer tsID : data.docs[docID].tss) {
                            Cmn[docToPartitionBoTA1.get(docID)][tsToPartitionBoTA1.get(tsID)].incrementAndGet();
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
                if (maxCmn[col] < Cmn[row][(row + col) % P].get()) { // Correct.
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
     * Partition TS, divide by counting number of tss.
     * TSs are not necessarily consecutive. 
     * 
     * Result are Doc for TS partition and TS partition hash map. 
     * Resulted HashMap is newly created.
     * 
     * @throws Exception 
     */
    private void evenPartitionTS() throws Exception {
        docToPartitionBoTA1 = new HashMap<>(2 * M, 1f);
        tsToPartitionBoTA1 = new HashMap<>(2 * O, 1f);
        
        for (int i = 0; i < P; i++) {
            // Get range of doc for each partition.
            int docStart = i * M / P;
            int docEnd = (i + 1) * M / P;
            if (docEnd > M) {
                docEnd = M;
            }
            // Get range of ts for each partition.
            int tsStart = i * O / P;
            int tsEnd = (i + 1) * O / P;
            if (tsEnd > O) {
                tsEnd = O;
            }
            
            // Put word and doc in the specified range to partition map.
            for (int j = docStart; j < docEnd; j++) {
                docToPartitionBoTA1.put(docIDListBoTA1.get(j), i);
            }
            // Put ts in the specified range to partition map.
            for (int j = tsStart; j < tsEnd; j++) {
                tsToPartitionBoTA1.put(tsIDListBoTA1.get(j), i);
            }
        }
    }

    /**
     * Based on algorithm in GPU paper, divide by number of word instances in rows and columns. 
     * TSs are not necessarily consecutive. 
     * 
     * Result are Doc for TS partition and TS partition hash map. 
     * Resulted HashMap is newly created.
     * 
     * @throws Exception 
     */
    private void gpuPartitionTSParallel() throws Exception {
        docToPartitionBoTA1 = new HashMap<>(2 * M, 1f);
        tsToPartitionBoTA1 = new HashMap<>(2 * O, 1f);

        // Total number Word Instance.
        final AtomicLong S = new AtomicLong();
        // Number of Word Instance in each doc and of each word.
        final AtomicLong[] docInstance = new AtomicLong[M];
        final AtomicLong[] tsInstance = new AtomicLong[O];
        for (int i = 0; i < M; i++) {
            docInstance[i] = new AtomicLong();
        }
        for (int i = 0; i < O; i++) {
            tsInstance[i] = new AtomicLong();
        }
        
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);
        for (int m = 0; m < M; m++) {
            final int docID = m;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        // Count ts instances.
                        S.addAndGet(data.docs[docID].L);
                        docInstance[docID].set(data.docs[docID].L);
                        for (Integer tsID : data.docs[docID].tss) {
                            tsInstance[tsID].incrementAndGet();
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
        long sumInstance = 0;
        double goal = Copt;
        double gap = Copt;
        double oldGap;
        for (int i = 0; i < M; i++) {
            // This for loop divides documents into P even groups, 
            // with regard to number of word instances in each document.
            int docID = docIDListBoTA1.get(i);
            sumInstance += docInstance[docID].get();
            
            // Keep info about old gap.
            oldGap = gap;
            // Compute new gap for new document.
            gap = Math.abs(goal - sumInstance);
            if (gap > oldGap) {
                // Move to new partition.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S.get();
                }
                gap = Math.abs(goal - sumInstance);
            }
            
            docToPartitionBoTA1.put(docID, part);
        }

        part = 0;
        sumInstance = 0;
        goal = Copt;
        gap = Copt;
        for (int i = 0; i < O; i++) {
            // This for loop divides ts into P even groups, 
            // with regard to number of instances of each ts.
            int tsID = tsIDListBoTA1.get(i);
            sumInstance += tsInstance[tsID].get();
            
            // Keep info about old gap, gap always may change after each loop over new ts.
            oldGap = gap;
            // Compute new gap for new ts.
            gap = Math.abs(goal - sumInstance);
            if (gap > oldGap) {
                // Move to new partition. The last ts would still be in partition P-1.
                part++;
                // Recompute goal and gap, so in the next iteration old gap will be the largest gap.
                if (part < P - 1) {
                    goal = (part + 1) * Copt;
                } else {
                    goal = S.get();
                }
                gap = Math.abs(goal - sumInstance);
            }
            
            tsToPartitionBoTA1.put(tsID, part);
        }
    }

    /**
     * Sync replicated data.
     * @throws Exception 
     */
    public void syncTTSSum() throws Exception {
        computeTTSSumMatrix();
        replicateTTSSumMatrix();
    }

    /**
     * Recompute T matrix based on all replication of it.
     * @throws Exception 
     */
    private void computeTTSSumMatrix() throws Exception {
        int[] newT = new int[K];
        
        for (int k = 0; k < K; k++) {
            for (int p = 0; p < P; p++) {
                newT[k] += ntssumList[p][k] - ntssum[k];
            }
            newT[k] += ntssum[k];
        }
        
        ntssum = newT;
    }
    
    /**
     * Make replication of T matrix.
     * @throws Exception 
     */
    private void replicateTTSSumMatrix() throws Exception {
        ntssumList = new int[P][];
        for (int i = 0; i < P; i++) {
            // .clone() or Arrays.copyOf() will deep copy to 1 level.
            // That is, it works for 1-D array, not 2-D array or ArrayList of object.
            ntssumList[i] = Arrays.copyOf(ntssum, K);
        }
    }

    public void computePi() {
        for (int k = 0; k < K; k++) {
            for (int ts = 0; ts < O; ts++) {
                pi[k][ts] = (float) ((nts[ts][k] + gamma) / (ntssum[k] + O * gamma));
            }
        }
    }

    public void computeCumulativePi() {
        for (int k = 0; k < K; k++) {
            for (int ts = 0; ts < O; ts++) {
                stationaryPi[k][ts] += pi[k][ts];
            }
        }
    }

    public void computeStationaryPi() {
        for (int k = 0; k < K; k++) {
            for (int ts = 0; ts < O; ts++) {
                stationaryPi[k][ts] /= numCumulativeSample;
            }
        }
    }

    /**
     * Init parameters and read data for estimation.
     * 
     * @param option
     * @return 
     * @throws java.lang.Exception
     */
    public boolean initNewModelBoT(PLDACmdOption option) throws Exception {
        super.initNewModel(option);
        shuffleTimesTS = option.shuffleTimesTS;
        L = option.L;
        
        // Transfer data from old dataset to new dataset.
        this.data = new LDADataset_BoT(super.data);
        if (super.dataTest != null) {
            this.dataTest = new LDADataset_BoT(super.dataTest);
        }
        super.data = this.data;
        super.dataTest = this.dataTest;
        
        // BoT specific init.
        gamma = option.gamma;
        if (gamma < 0) {
            gamma = 0.1;
        }
        
        // Read time data.
        readTSData(option);
        O = this.data.O;

        // Timestamp variable init.
        nts = new int[O][K];
        for (int ts = 0; ts < O; ts++) {
            for (int k = 0; k < K; k++) {
                nts[ts][k] = 0;
            }
        }

        ntssum = new int[K];
        for (int k = 0; k < K; k++) {
            ntssum[k] = 0;
        }

        y = new short[M][];
        for (int m = 0; m < data.M; m++) {
            int L = data.docs[m].L;
            y[m] = new short[L];
            //initilize for y
            for (int i = 0; i < L; i++) {
                int topic = (int) Math.floor(Math.random() * K);
                y[m][i] = (short) topic;
                // number of ts of i assigned to topic j
                nts[data.docs[m].tss[i]][topic] += 1;
                // number of tokens in document i assigned to topic j
                nd[m][topic] += 1;
                // total number of ts assigned to topic j
                ntssum[topic] += 1;
            }
            // total number of words in document i
            ndsum[m] += L;
        }

        pi = new float[K][O];
        
        if (option.howToGetDistribution == 2) {
            stationaryPi = new float[K][O];
        }

        return true;
    }
    
    /**
     * Already read data, now read timestamp data.
     * Read ts data for existing dataset.
     * 
     * @param option
     * @return true if read successfully.
     * @throws Exception 
     */
    public boolean readTSData(PLDACmdOption option) throws Exception {
        if (this.data != null) {
            if (option.tsfileFormat.trim().equalsIgnoreCase("Single")) {
                LDADataset_BoT.readTSDataMASSingle(this.data, dir + File.separator + option.tsfile, option.L);
            } else if (option.tsfileFormat.trim().equalsIgnoreCase("Array")) {
                LDADataset_BoT.readTSDataArray(this.data, dir + File.separator + option.tsfile);
            } else if (option.tsfileFormat.trim().equalsIgnoreCase("CitRef")) {
                LDADataset_BoT.readTSDataCitRef(this.data, dir + File.separator + option.tsfile);
            }
            // Save timestamp map right after reading from datafile.
            this.data.localDictBoT.writeWordMap(option.dir + File.separator + "TSMap_" + option.dfile);
        } else {
            return false;
        }
        
        if (this.dataTest != null) {
            this.dataTest.O = this.data.O;
            for (int i = 0; i < M; i++) {
                this.dataTest.docs[i].L = this.data.docs[i].L;
                this.dataTest.docs[i].tss = this.data.docs[i].tss;
            }
        }

        return true;
    }

    /**
     * Save model, with new var: 
     * 
     * @param modelName
     * @return 
     */
    @Override
    public boolean saveModel(String modelName) {
        super.saveModel(modelName);

        if (!saveModelTSTAssign(dir + File.separator + modelName + tstassignSuffix)) {
            return false;
        }
        
        if (!saveModelTTSs(dir + File.separator + modelName + ttssSuffix)) {
            return false;
        }
        
        if (!saveModelPi(dir + File.separator + modelName + piSuffix)) {
            return false;
        }
        
        if (!saveModelStationaryPi(dir + File.separator + modelName + stationaryPiSuffix)) {
            return false;
        }
        
        return true;
    }

    /**
     * Save TS-topic assignments for this model.
     * The model is learn for train data, not for test data.
     */
    public boolean saveModelTSTAssign(String filename) {
        int i, j;

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

            //write docs with topic assignments for words
            for (i = 0; i < data.M; i++) {
                for (j = 0; j < data.docs[i].L; ++j) {
                    writer.write(data.docs[i].tss[j] + ":" + y[i][j] + " ");
                }
                writer.write("\n");
            }

            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving model tstassign: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Save the most likely TSs for each topic.
     * The model is learn for train data, not for test data.
     * 
     * Share option twords, the number of top tokens to save for each topic, with method saveModelTWords.
     */
    public boolean saveModelTTSs(String filename) {
        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(filename), "UTF-8"));

            if (twords > O) {
                twords = O;
            }

            for (int k = 0; k < K; k++) {
                List<Pair> tssProbsList = new ArrayList<Pair>();
                for (int tsID = 0; tsID < O; tsID++) {
                    Pair p = new Pair(tsID, pi[k][tsID], false); // naturalorder == false, sort descending.
                    tssProbsList.add(p);
                }//end foreach word

                //print topic				
                writer.write("Topic " + k + "th:\n");
                Collections.sort(tssProbsList);

                for (int i = 0; i < twords; i++) {
                    if (data.localDictBoT.contains((Integer) tssProbsList.get(i).first)) {
                        String ts = data.localDictBoT.getWord((Integer) tssProbsList.get(i).first);

                        writer.write("\t" + ts + " " + tssProbsList.get(i).second + "\n");
                    }
                }
            } //end foreach topic			

            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving model ttss: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean saveModelPi(String filename) {
        if (pi == null) {
            return true;
        }
        
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < O; j++) {
                    writer.write(pi[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving Pi: TS topic distribution file for this model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean saveModelStationaryPi(String filename) {
        if (stationaryPi == null) {
            return true;
        }
        
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < O; j++) {
                    writer.write(stationaryPi[i][j] + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Error while saving stationary Pi: TS topic distribution file for this model: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Algorithm 1 to permute DTS matrix.
     * 
     * @throws Exception 
     */
    private void dtsPermuteParallel() throws Exception {
        // Count into array number of Word Instance in each doc and of each word.
        final AtomicLong[] docTSInstance = new AtomicLong[M];
        final AtomicLong[] tsTSInstance = new AtomicLong[O];
        for (int i = 0; i < M; i++) {
            docTSInstance[i] = new AtomicLong();
        }
        for (int i = 0; i < O; i++) {
            tsTSInstance[i] = new AtomicLong();
        }
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);
        for (int m = 0; m < M; m++) {
            final int docID = m;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        // Counting.
                        docTSInstance[docID].set(data.docs[docID].L);
                        for (Integer tsID : data.docs[docID].tss) {
                            tsTSInstance[tsID].incrementAndGet();
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
        
        // Convert array to hashmap to keep id information before sorting.
        LinkedHashMap<Integer, Long> docTSInstanceHM = GeneralUtility.arrayToLinkedHashMap(docTSInstance);
        LinkedHashMap<Integer, Long> tsTSInstanceHM = GeneralUtility.arrayToLinkedHashMap(tsTSInstance);
        
        // Sort hashmap.
        docTSInstanceHM = GeneralUtility.getSortedMapDescending(docTSInstanceHM);
        tsTSInstanceHM = GeneralUtility.getSortedMapDescending(tsTSInstanceHM);
        
        // Get sorted id list
        docIDListBoTA1 = new ArrayList(docTSInstanceHM.keySet());
        tsIDListBoTA1 = new ArrayList(tsTSInstanceHM.keySet());

        // Run copy permuted id list (interpose most-least).
        if (this.permuteAlgorithm.startsWith("A1H1")) {
            docIDListBoTA1 = GeneralUtility.interposeList(docIDListBoTA1);
            tsIDListBoTA1 = GeneralUtility.interposeList(tsIDListBoTA1);
        } else if (this.permuteAlgorithm.startsWith("A1H2")) {
            docIDListBoTA1 = GeneralUtility.interposeSymmetryList(docIDListBoTA1);
            tsIDListBoTA1 = GeneralUtility.interposeSymmetryList(tsIDListBoTA1);
        } else if (this.permuteAlgorithm.startsWith("A2")) {
            docIDListBoTA1 = GeneralUtility.rangerList(docIDListBoTA1, P);
            tsIDListBoTA1 = GeneralUtility.rangerList(tsIDListBoTA1, P);
        }
        
        // Run symmetric transformation (swap).
        if (this.permuteAlgorithm.endsWith("A")) {
            // Keep current order.
        } else if (this.permuteAlgorithm.endsWith("B")) {
            // Swap vertically: swap word.
            GeneralUtility.swapList(tsIDListBoTA1);
        } else if (this.permuteAlgorithm.endsWith("C")) {
            // Swap vertically: swap both doc and word.
            GeneralUtility.swapList(docIDListBoTA1);
            GeneralUtility.swapList(tsIDListBoTA1);
        } else if (this.permuteAlgorithm.endsWith("D")) {
            // Swap vertically: swap doc.
            GeneralUtility.swapList(docIDListBoTA1);
        }
    }
}