/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.Algorithm;

import cgs_lda_multicore.DataModel.PModel;
import cgs_lda_multicore.DataModel.PModel_BoT;
import cgs_lda_multicore.Utility.PLDACmdOption;
import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import jgibblda.Conversion;
import jgibblda.Estimator;

/**
 * Learn LDA Model for a dataset using CGS LDA.
 * 
 * The result is the sample of the model:
 * 1. Topic assignment z of ALL word instance
 * -> 2. Count to WT, DT...
 * -> 3. Theta (D-T distribution), Phi (T-W distribution)
 * => These 3 results will converge to stationary.
 * Given Stationary pi and Transition tau, we have pi*tau = pi
 * This is when then statistics do not change much between iterations, many ways to check convergence.
 * 
 * However, even when converged they will fluctuate around the true distribution.
 * => Need to get average statistics from samples when computing distribution. Several approaches:
 * A1: Average of counting of samples (e.g., 10) from 1 chain, with thinning interval: like PGM course.
 * A2: Average of final distribution theta and phi from different chains: like Newman 2007.
 * A3: Just average final distribution theta and phi from different samples in 1 chain, 
 * with thinning interval (this is different from A1, because Theta and Phi is not linear to counting).
 * => Actually, the fluctuation is very small, so we can ignore it, and compute Theta and Phi from
 * only 1 sample.
 * 
 * Modify:
 * The old code implementation is not averaged, it only compute from 1 sample.
 * In this new code, we use A3: average the final distribution theta and phi 
 * from 1 chain with thinning interval.
 * 
 * But the size of Theta is too large: so only compute when needed, specify by flag how to compute distribution.
 * 
 * @author THNghiep
 */
public class PEstimator_BoT extends PEstimator {
    /**
     * Train model using PModel
     */
    public PModel_BoT trnModel;

    /**
     * Init using POption.
     * 
     * @param option
     * @return
     * @throws Exception 
     */
    @Override
    public boolean init(PLDACmdOption option) throws Exception {
        this.option = option;
        this.trnModel = new PModel_BoT();
        super.trnModel = this.trnModel;
        
        if (option.est) {
            // Estimate from scratch.
            // Note for Java polymorphism:
            // - Subclass param is auto converted to super class.
            // - When there are 2 suitable method in subclass and super class, priority go for current class instance.
            // That is PModel_BoT.initNewModel(PLDACmdOption).
            if (!trnModel.initNewModelBoT(option)) {
                // Only this method has implemented test set.
                return false;
            }
            trnModel.partitionData(this.option.howToPartition);
            trnModel.partitionTS(this.option.howToPartition);
        }

        return true;
    }

    /**
     * Estimate in parallel.
     * BoTA1 is the original paper's BoT model. 
     * Other extensions to BoT model are planned, such as vary-length time array, citation/reference time array.
     * 
     * @throws Exception 
     */
    public void estimateParallelGPUAlgorithm_BoTA1() throws Exception {
        System.out.println("Sampling " + trnModel.niters + " iteration more.");
        System.out.println("Parallel on " + trnModel.P + " partitions. Thread pool size: " + trnModel.threadPoolSize);
        System.out.println("Total " + trnModel.O + " timestamps. Timestamp array size: " + trnModel.L);
        int P = trnModel.P;
        int threadPoolSize = trnModel.threadPoolSize;
        
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);

        // Continue from saved last iteration.
        int lastIter = trnModel.liter;
        
        System.out.println("Start iteration " + (trnModel.liter + 1) + " ...");
        long startAll = System.currentTimeMillis();
        for (trnModel.liter = lastIter + 1; trnModel.liter <= lastIter + trnModel.niters; trnModel.liter++) {
            long start = System.currentTimeMillis();
            for (int col = 0; col < P; col++) {
                // Sampling step.
                // Note that: 
                // - Sampling word would be based on DT, TW, and TW sum.
                // - Sampling ts would be based on DT, TTS, and TTS sum.
                // => so do not need to syncTWSum T between sampling word and ts, but need to finish on DT.
                
                // First turn: sampling word.
                // Reset counting finished thread.
                countThreadFinish.set(0);
                for (int row = 0; row < P; row++) {
                    // All row in parallel, so row is also partitionID.
                    final int realRow = row;
                    final int realCol = (row + col) % P;
                    executor.submit(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                // Call for each thread.
                                estimatePartition(realRow, realCol);
                                int threadNum = countThreadFinish.incrementAndGet();
                                //System.out.println("Finished thread " + threadNum);
                            } catch (Exception ex) {
                                System.err.println(ex.toString());
                                ex.printStackTrace();
                            }
                        }
                    });
                }
                // Wait until all thread finished.
                // When all thread is finished, all local cached data are flushed, so no Visibility problem with multithreading.
                while (countThreadFinish.get() != P) {}
                trnModel.syncTWSum();
                
                // Second turn: After finised sampling word on DT, Sampling timestamp.
                countThreadFinish.set(0);
                for (int row = 0; row < P; row++) {
                    // All row in parallel, so row is also partitionID.
                    final int realRow = row;
                    final int realCol = (row + col) % P;
                    executor.submit(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                // Call for each thread.
                                estimatePartitionTS(realRow, realCol);
                                int threadNum = countThreadFinish.incrementAndGet();
                                //System.out.println("Finished thread " + threadNum);
                            } catch (Exception ex) {
                                System.err.println(ex.toString());
                                ex.printStackTrace();
                            }
                        }
                    });
                }
                // Wait until all thread finished.
                // When all thread is finished, all local cached data are flushed, so no Visibility problem with multithreading.
                while (countThreadFinish.get() != P) {}
                trnModel.syncTTSSum();
            }
            // Time each iteration.
            long elapse = System.currentTimeMillis() - start;
            System.out.println("Finish iteration " + trnModel.liter + " in " + (double) elapse / 1000 + " seconds. (P = " + trnModel.P + ", threadPoolSize = " + trnModel.threadPoolSize + ")");

            if ((trnModel.liter > option.burnIn) && (option.savestep > 0)) {
                if (trnModel.liter % option.savestep == 0) { // save in each thinning interval
                    System.out.println("Saving the model at iteration " + trnModel.liter + " ...");
                    trnModel.computeTheta();
                    trnModel.computePhi();
                    trnModel.computePi();
                    if (option.howToGetDistribution == 2) {
                        trnModel.computeCumulativeTheta();
                        trnModel.computeCumulativePhi();
                        trnModel.computeCumulativePi();
                        trnModel.numCumulativeSample++;
                    }
                    trnModel.saveModel("model-" + Conversion.ZeroPad(trnModel.liter, 5));
                }
            }
        }// End sampling iterations.
        // Shutdown thread pool, wait until shutdown finished.
        executor.shutdown();
        while (!executor.isTerminated()) {}
        
        // Time all iterations.
        long elapseAll = System.currentTimeMillis() - startAll;
        System.out.println("Finish " + trnModel.niters + " iterations in " + (double) elapseAll / 1000 + " seconds. (P = " + trnModel.P + ", threadPoolSize = " + trnModel.threadPoolSize + ")");

        System.out.println("Gibbs sampling completed.");
        System.out.println("Saving the final model.");
        trnModel.computeTheta();
        trnModel.computePhi();
        trnModel.computePi();
        if (option.howToGetDistribution == 2) {
            trnModel.computeStationaryTheta();
            trnModel.computeStationaryPhi();
            trnModel.computeStationaryPi();
        }
        trnModel.liter--;
        trnModel.saveModel("model-final");
    }
    
    /**
     * Estimate TS for 1 partition.
     * 
     * @param row
     * @param col
     */
    public void estimatePartitionTS(int row, int col) throws Exception {
//        System.out.println("Start Sampling TS: Partition row: " + row + ", col: " + col + ".");
        
        // loop over every ts instance assignment z_i.
        // y is a type of var: the topic assignment. There are many y var: each assignment.
        for (int docID : trnModel.partitionToDocBoTA1.get(row)) {
            for (int i = 0; i < trnModel.data.docs[docID].L; i++) {
                int tsID = trnModel.data.docs[docID].tss[i];
                if (trnModel.partitionToTSBoTA1.get(col).contains(tsID)) {
                    // y_i = y[docID][i]
                    // Now sample y_i from p(y_i|y_-i, ts)
                    int topic = samplingTS(row, docID, i);
                    trnModel.y[docID][i] = (short) topic;
                }
            }// end for each word
        }// end for each document

//        System.out.println("Finish Sampling TS: Partition row: " + row + ", col: " + col + ".");
    }

    /**
     * Sampling in a specific partition.
     * Notice: possible negative value when many thread decrease 1 matrix?
     * --> No, because non-conflict on DT, TW and copy-sync on T. 
     * With full sync on T, T is consistent with Z, so no negative.
     * 
     * @param partID
     * @param docID
     * @param i
     * @return topic assignment.
     */
    public int samplingTS(int partID, int docID, int i) throws Exception {
        // remove y_i from the count variable
        int topic = trnModel.y[docID][i];
        int tsID = trnModel.data.docs[docID].tss[i];

        // Parallel access to these arrays, no conflict -> no race condition or visibility problem, no lock.
        trnModel.nts[tsID][topic] -= 1; // WT
        trnModel.nd[docID][topic] -= 1; // DT
        trnModel.ntssumList[partID][topic] -= 1; // TTS
        trnModel.ndsum[docID] -= 1; // doc length

        double Ogamma = trnModel.O * trnModel.gamma;
        double Kalpha = trnModel.K * trnModel.alpha;

        // Do multinominal sampling via cumulative method.
        // Also divide by doc length because not normalize later.
        // Meaning: P(ts_assignment=topic) = P(ts|topic) * P(topic|doc)
        for (int k = 0; k < trnModel.K; k++) {
            trnModel.pList[partID][k] = 
                    (trnModel.nts[tsID][k] + trnModel.gamma) / (trnModel.ntssumList[partID][k] + Ogamma) 
                    * (trnModel.nd[docID][k] + trnModel.alpha) / (trnModel.ndsum[docID] + Kalpha);
        }

        // cumulate multinomial parameters
        for (int k = 1; k < trnModel.K; k++) {
            trnModel.pList[partID][k] += trnModel.pList[partID][k - 1];
        }

        // Uniform random. Scaled sample to match unnormalized p[].
        double u = Math.random() * trnModel.pList[partID][trnModel.K - 1];

        // sample topic w.r.t distribution p, remember topic
        for (topic = 0; topic < trnModel.K; topic++) {
            if (trnModel.pList[partID][topic] > u) {
                break;
            }
        }

        // add newly estimated z_i to count variables
        trnModel.nts[tsID][topic] += 1;
        trnModel.nd[docID][topic] += 1;
        trnModel.ntssumList[partID][topic] += 1;
        trnModel.ndsum[docID] += 1;

        return topic;
    }
}
