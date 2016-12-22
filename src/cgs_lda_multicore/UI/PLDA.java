/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.UI;

import cgs_lda_multicore.Algorithm.PEstimator;
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
public class PLDA {
    public static void main(String args[]) {
        PLDACmdOption option = new PLDACmdOption();
        CmdLineParser parser = new CmdLineParser(option);

        try {
            if (args.length == 0) {
                showHelp(parser);
                return;
            }

            parser.parseArgument(args);

            /*// If only 1 thread: not parallel, use old code.
            if (option.P == 1) {
                LDA.main(args);
                return;
            }*/
            
            // If multithread.
            if (option.est || option.estc) {
                PEstimator pEstimator = new PEstimator();
                
                // Init, including partition data.
                long start = System.currentTimeMillis();
                pEstimator.init(option);
                long elapse = System.currentTimeMillis() - start;
                System.out.println("Model init time: " + (double) elapse / 1000 + " second.");
                
                // Print eta.
                System.out.println("Best eta: " + pEstimator.trnModel.etaBest);
                System.out.println("Average random eta: " + pEstimator.trnModel.etaRandom);
                System.out.println("Press enter to continue.");
                System.in.read();

                // Estimate with parallel processing.
                start = System.currentTimeMillis();
                pEstimator.estimateParallelGPUAlgorithm();
                elapse = System.currentTimeMillis() - start;
                System.out.println("Model training time: " + (double) elapse / 1000 + " second.");

                double perplexity = 0;
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
                
            } else if (option.inf) {
                PInferencer pInferencer = new PInferencer();
                pInferencer.init(option);
                pInferencer.inferenceParallelGPUAlgorithm();

                for (int i = 0; i < pInferencer.newModel.phi.length; ++i) {
                    //phi: K * V
                    System.out.println("-----------------------\nTopic " + i + ":");
                    for (int j = 0; j < 10; ++j) {
                        System.out.println(pInferencer.globalDict.id2word.get(j) + "\t" + pInferencer.newModel.phi[i][j]);
                    }
                }
                System.out.println("See saved model for full Inference result.");
                
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
        System.out.println("Parallel Topic Modeling [options...] [arguments...]");
        parser.printUsage(System.out);
    }

}
