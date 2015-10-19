/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.Utility;

import cgs_lda_multicore.DataModel.PModel;
import jgibblda.LDADataset;

/**
 *
 * @author THNghiep
 */
public class LDAUtility {
    /**
     * Compute perplexity for a test set based on a trained model.
     * 
     * Based on formula in Newman et al., 2009.
     * 
     * Here, the test set have the same documents and vocabulary as data in model.
     * Test set is divided from training set so that word instances in each document 
     * are randomly selected.
     * Note: This method only computes perplexity from 1 chain (in provided model).
     * 
     * @param model
     * @param data if null, get data from model.
     * @param howToComputeDistribution 1: Theta and Phi from final sample, 
     * 2: Stationary Theta and Phi averaged from all saved Theta and Phi.
     * Note for experiment: Check empirically which way howToComputeDistribution get smaller perplexity.
     * 
     * @return perplexity of test set.
     * @throws Exception 
     */
    public static double computePerplexitySingleChain(PModel model, LDADataset data, 
            int howToComputeDistribution) throws Exception {
        if (data == null) {
            data = model.data;
        }
        float[][] theta = null;
        float[][] phi = null;
        if (howToComputeDistribution == 1) {
            theta = model.theta;
            phi = model.phi;
        } else if (howToComputeDistribution == 2) {
            theta = model.stationaryTheta;
            phi = model.stationaryPhi;
        }
        
        double perplexity = 0;
        
        double loglikelihood = 0;
        long Ntest = 0;

        for (int m = 0; m < data.M; m++) {
            Ntest += data.docs[m].length;
            for (int n = 0; n < data.docs[m].length; n++) {
                // For each word instance (doc-word pair) in data test set.
                // This iterating scheme is equivalent to formula in Newman et al. 2009, 
                // as they loop over each unique pair doc-word, with each pair they multiply by number of duplication in each doc.
                double sum = 0;
                for (int k = 0; k < model.K; k++) {
                    // compute inner product.
                    // document and word ID from model and test dataset have to be matched.
                    // that is, test set and training set are divided from the same dataset.
                    sum += theta[m][k] * phi[k][data.docs[m].words[n]];
//                    if (theta[m][k] * phi[k][data.docs[m].words[n]] == 0) {
//                        System.out.println("m, w, k: " + m + ", " + data.docs[m].words[n] + ", " + k);
//                    }
                }
                loglikelihood += Math.log(sum);
//                if (sum == 0) {
//                    System.out.println("m, w: " + m + ", " + data.docs[m].words[n]);
//                }
            }
        }
//        System.out.println("logli, V: " + loglikelihood + ", " + data.V);
        perplexity = Math.exp(-loglikelihood / Ntest);

        return perplexity;
    }
}
