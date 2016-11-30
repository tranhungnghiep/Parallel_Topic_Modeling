/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.Utility;

import jgibblda.LDACmdOption;
import org.kohsuke.args4j.Option;

/**
 *
 * @author THNghiep
 */
public class PLDACmdOption extends LDACmdOption {
    @Option(name = "-burnin", usage = "Burn in period: number of samples (iterations) to discard at the beginning of MC. Default = 200")
    public int burnIn = 200;

    @Option(name = "-P", usage = "Number of partitions. Default = num of cores - 1.")
    public int P = GeneralUtility.getNumOfCore() - 1;

    @Option(name = "-threadpoolsize", usage = "Maximum number of threads to run in parallel. To take advantage of parallel CGS LDA and partitioning algorithm, it should be: threadpoolsize >= P. Default in PModel: threadpoolsize = P (num of partitions).")
    public int threadPoolSize = 0;

    @Option(name = "-shuffle", usage = "Number of shuffles. Default = 10.")
    public int shuffleTimes = 10;

    @Option(name = "-howtopart", usage = "How to partition: 1: even, 2: gpu. Default = 2.")
    public int howToPartition = 2;
    
    @Option(name = "-howtogetdist", usage = "How to get distribution of Theta and Phi: 1: using final sample, 2: average of theta and phi from many samples. Default = 1.")
    public int howToGetDistribution = 1;
    
    @Option(name = "-testsetprop", usage = "Proportion of test set from dataset. Default = 0.1.")
    public double testSetProportion = 0.1;

    @Option(name = "-dfiletrain", usage = "Specify training data file")
    public String dfiletrain = "";

    @Option(name = "-dfiletest", usage = "Specify test data file")
    public String dfiletest = "";

    @Option(name = "-isseptestset", usage = "Separate training set and test set or not. Default = false.")
    public boolean isSepTestset = false;

    @Option(name = "-datafileformat", usage = "Format of datafile input. Value = {Private, NYT, CORE, ...}. Default = private format.")
    public String datafileFormat = "Private";
}