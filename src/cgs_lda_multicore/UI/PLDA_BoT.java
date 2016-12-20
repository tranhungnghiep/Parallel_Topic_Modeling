/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.UI;

import cgs_lda_multicore.Algorithm.PEstimator;
import cgs_lda_multicore.Algorithm.PEstimator_BoT;
import cgs_lda_multicore.Algorithm.PInferencer;
import cgs_lda_multicore.Utility.LDAUtility;
import cgs_lda_multicore.Utility.PLDACmdOption;
import jgibblda.LDA;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

/**
 *
 * @author THNghiep
 */
public class PLDA_BoT {
    public static void main(String args[]) {
        PLDACmdOption option = new PLDACmdOption();
        CmdLineParser parser = new CmdLineParser(option);

        try {
            if (args.length == 0) {
                showHelp(parser);
                return;
            }

            parser.parseArgument(args);

            if (option.est || option.estc) {
                PEstimator_BoT pEstimator = new PEstimator_BoT();
                
                // Init, including partition data.
                long start = System.currentTimeMillis();
                pEstimator.init(option);
                long elapse = System.currentTimeMillis() - start;
                System.out.println("Model init time: " + (double) elapse / 1000 + " second.");
                pEstimator.trnModel.data.printDatasetStatistics();
                pEstimator.trnModel.data.printDatasetUnitTestInfo(100000); // For testing purpose: check if paper and timestamp synced.
                
                // Print eta.
                System.out.println("Best eta: " + pEstimator.trnModel.etaBest);
                System.out.println("Average random eta: " + pEstimator.trnModel.etaRandom);
                System.out.println("Best eta BoT: " + pEstimator.trnModel.etaTSBestBoTA1);
                System.out.println("Average random eta BoT: " + pEstimator.trnModel.etaTSRandomBoTA1);
                System.out.println("Press enter to continue.");
                System.in.read();

                // Estimate with parallel processing.
                start = System.currentTimeMillis();
                pEstimator.estimateParallelGPUAlgorithm_BoTA1();
                elapse = System.currentTimeMillis() - start;
                System.out.println("Model training time: " + (double) elapse / 1000 + " second.");

                double perplexity = 0;
                // Note: Currently Perplexity is only computed for word, not for timestamp.
                if (pEstimator.trnModel.dataTest == null) {
                    // Compute perplexity of train data.
                    perplexity = LDAUtility.computePerplexitySingleChain(pEstimator.trnModel, null, option.howToGetDistribution);
                    System.out.println("Perplexity of train data (lower is better): " + perplexity);
                } else {
                    // Compute perplexity of separated test data.
                    perplexity = LDAUtility.computePerplexitySingleChain(pEstimator.trnModel, pEstimator.trnModel.dataTest, option.howToGetDistribution);
                    System.out.println("Perplexity of separated test data (lower is better): " + perplexity);
                }

                System.out.println("See saved model for the Estimating result.");
            }
        } catch (CmdLineException cle) {
            System.out.println("Command line error: " + cle.getMessage());
            showHelp(parser);
        } catch (Exception e) {
            System.out.println("Error in main: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void showHelp(CmdLineParser parser) {
        System.out.println("CGS LDA Multicore [options...] [arguments...]");
        parser.printUsage(System.out);
    }

}
