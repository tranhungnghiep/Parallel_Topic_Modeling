/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.Algorithm;

import cgs_lda_multicore.DataModel.PModel;
import cgs_lda_multicore.Utility.PLDACmdOption;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import jgibblda.Conversion;
import jgibblda.Model;

/**
 * Train a New model from new data, based on prior from trained model.
 * This is a special way to enhance quality of model on small data, based on trained model on other large data.
 * 
 * @author THNghiep
 */
public class PInferencer extends jgibblda.Inferencer {
    static AtomicInteger countThreadFinish;

    /**
     * New model: main model, train from new data, based on prior from trained model.
     */
    public PModel newModel;

    private PLDACmdOption option;

    /**
     * Init method.
     * 
     * @param option
     * @return 
     */
    public boolean init(PLDACmdOption option) throws Exception {
        this.option = option;
        
        System.out.println("Init trained model.");
        trnModel = new Model();
        if (!trnModel.initEstimatedModel(option)) {
            return false;
        }

        globalDict = trnModel.data.localDict;
        computeTrnTheta();
        computeTrnPhi();

        System.out.println("Init new model.");
        newModel = new PModel();
        // Init new model based on trained model: copy alpha, beta, K.
        if (!newModel.initNewModel(option, trnModel)) {
            return false;
        }
        newModel.partitionData(option.howToPartition);

        return true;
    }

    /**
     * Inference new model, getting data from a specified dataset.
     * 
     * @throws Exception 
     */
    public void inferenceParallelGPUAlgorithm() throws Exception {
        newModel.niters = niters;
        System.out.println("Sampling " + newModel.niters + " iteration for inference.");
        System.out.println("Parallel on " + newModel.P + " partitions. Thread pool size: " + newModel.threadPoolSize);
        int P = newModel.P;
        int threadPoolSize = newModel.threadPoolSize;
        
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(threadPoolSize);

        for (newModel.liter = 1; newModel.liter <= newModel.niters; newModel.liter++) {
            System.out.println("Iteration " + newModel.liter + " ...");
            for (int col = 0; col < P; col++) {
                // Sampling step.
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
                                inferencePartition(realRow, realCol);
                                int threadNum = countThreadFinish.incrementAndGet();
                                //System.out.println("Finished thread " + threadNum);
                            } catch (Exception ex) {
                                System.err.println(ex.toString());
                            }
                        }
                    });
                }
                // Sync step.
                // Wait until all thread finished.
                // When all thread is finished, all local cached data are flushed, so no Visibility problem with multithreading.
                while (countThreadFinish.get() != P) {}
                newModel.syncTWSum();
            }

            if (option.savestep > 0) {
                if (newModel.liter % option.savestep == 0) { // save in each thinning interval
                    System.out.println("Saving the model at iteration " + newModel.liter + " ...");
                    newModel.computeTheta();
                    newModel.computeNewPhi(trnModel);
                    newModel.computeCumulativeTheta();
                    newModel.computeCumulativePhi();
                    newModel.numCumulativeSample++;
                    newModel.saveModel("model-" + Conversion.ZeroPad(newModel.liter, 5));
                }
            }
        }// End inference iterations.
        // Shutdown thread pool, wait until shutdown finished.
        executor.shutdown();
        while (!executor.isTerminated()) {}
        
        System.out.println("Gibbs sampling for inference completed.");
        System.out.println("Saving the final model.");
        newModel.computeTheta();
        newModel.computeNewPhi(trnModel);
        newModel.computeStationaryTheta();
        newModel.computeStationaryPhi();
        newModel.liter--;
        newModel.saveModel("model-final");
    }
    
    /**
     * Inference (or Estimate parameter for new model based on prior from old model) for 1 partition.
     * 
     * @param row
     * @param col
     */
    public void inferencePartition(int row, int col) {
        System.out.println("Start: Partition row: " + row + ", col: " + col + ".");
        
        // for all newz_i (word instance in new data).
        for (int docID : newModel.partitionToDoc.get(row)) {
            for (int i = 0; i < newModel.data.docs[docID].length; i++) {
                int wordID = newModel.data.docs[docID].words[i];
                if (newModel.partitionToWord.get(col).contains(wordID)) {
                    // newz_i = newz[docID][i]
                    // Now sample from p(z_i|z_-1,w)
                    int topic = infSampling(row, docID, i); // Similar to PEstimator class. However, newModel is learned with PRIOR from trainModel.
//                    newModel.z[docID].set(n, topic); // Array and ArrayList: no lock.
                    newModel.z[docID][i] = (short) topic; // Array and ArrayList: no lock.
                }
            }
        }

        System.out.println("Finish: Partition row: " + row + ", col: " + col + ".");
    }

    /**
     * Do sampling for inference.
     * 
     * @param partID
     * @param docID
     * @param i
     * @return new sampled topic.
     */
    protected int infSampling(int partID, int docID, int i) {
        // remove z_i from the count variables
//        int topic = newModel.z[docID].get(i);
        int topic = newModel.z[docID][i];
        int wordIDLocal = newModel.data.docs[docID].words[i];
        int wordIDGlobal = newModel.data.lid2gid.get(wordIDLocal);
        
        // Parallel access.
        newModel.nw[wordIDLocal][topic] -= 1;
        newModel.nd[docID][topic] -= 1;
//        newModel.nwsumList.get(partID)[topic] -= 1;
        newModel.nwsumList[partID][topic] -= 1;
        newModel.ndsum[docID] -= 1;

        // Using V and K from PRIOR trained model.
        double Vbeta = trnModel.V * newModel.beta;
        double Kalpha = trnModel.K * newModel.alpha;

        // Do multinomial sampling via cummulative method.
        // newModel.K == trainedModel.K
        for (int k = 0; k < newModel.K; k++) {
            newModel.pList[partID][k] = 
                    (trnModel.nw[wordIDGlobal][k] + newModel.nw[wordIDLocal][k] + newModel.beta) 
//                    / (trnModel.nwsum[k] + newModel.nwsumList.get(partID)[k] + Vbeta)
                    / (trnModel.nwsum[k] + newModel.nwsumList[partID][k] + Vbeta)
                    * (newModel.nd[docID][k] + newModel.alpha) 
                    / (newModel.ndsum[docID] + Kalpha);
        }

        // cummulate multinomial parameters
        for (int k = 1; k < newModel.K; k++) {
            newModel.pList[partID][k] += newModel.pList[partID][k - 1];
        }

        // scaled sample because of unnormalized p[]
        double u = Math.random() * newModel.pList[partID][newModel.K - 1];

        for (topic = 0; topic < newModel.K; topic++) {
//            if (newModel.pList.get(partID)[topic] > u) {
            if (newModel.pList[partID][topic] > u) {
                break;
            }
        }

        // add newly estimated z_i to count variables
        newModel.nw[wordIDLocal][topic] += 1;
        newModel.nd[docID][topic] += 1;
//        newModel.nwsumList.get(partID)[topic] += 1;
        newModel.nwsumList[partID][topic] += 1;
        newModel.ndsum[docID] += 1;

        return topic;
    }
}
