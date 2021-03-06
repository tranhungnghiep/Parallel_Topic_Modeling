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
    @Option(name = "-burnin", usage = "Burn in period: number of samples (iterations) to discard at the beginning of MC. Default = 100")
    public int burnIn = 100;

    @Option(name = "-P", usage = "Number of partitions. Default = num of cores - 1.")
    public int P = GeneralUtility.getNumOfCore() - 1;

    @Option(name = "-threadpoolsize", usage = "Maximum number of threads to run in parallel. To take advantage of parallel CGS LDA and partitioning algorithm, it should be: threadpoolsize >= P. Default in PModel: threadpoolsize = P (num of partitions).")
    public int threadPoolSize = 0;

    @Option(name = "-shuffle", usage = "Number of shuffles. Default = 10.")
    public int shuffleTimes = 10;

    @Option(name = "-shufflets", usage = "Number of shuffles TS. Default = 100.")
    public int shuffleTimesTS = 100;

    @Option(name = "-howtopart", usage = "How to partition: 1: even, 2: gpu. Default = 2.")
    public int howToPartition = 2;
    
    @Option(name = "-howtogetdist", usage = "How to get distribution of Theta and Phi: 1: using final sample, 2: average of theta and phi from many samples, require savestep > 0. Default = 1.")
    public int howToGetDistribution = 1;
    
    @Option(name = "-isseptestset", usage = "Data preparation: Separate training set and test set or not. Default = false.")
    public boolean isSepTestset = false;

    @Option(name = "-testsetprop", usage = "Data preparation: Proportion of test set from dataset to separate. Default = 0.1.")
    public double testSetProportion = 0.1;

    @Option(name = "-dfiletrain", usage = "Specify training data file")
    public String dfiletrain = "";

    @Option(name = "-dfiletest", usage = "Specify test data file")
    public String dfiletest = "";

    @Option(name = "-datafileformat", usage = "Format of datafile input. Value = {Private, NYT, CORE, ...}. Default = private format.")
    public String datafileFormat = "Private";

    @Option(name = "-gamma", usage = "Specify gamma")
    public double gamma = -1.0;

    @Option(name = "-tsfile", usage = "Specify timestamp data file")
    public String tsfile = "";

    /**
     * Single: e.g., each line is "timestamp"
     * Array: e.g., each line is "timestamp1 timestamp2"
     * CitRef: e.g., each line is "timestamp 0 cit_timestamp 0 ref_timestamp"
     */
    @Option(name = "-tsfileformat", usage = "Format of tsfile input. Value = {Single, Array, CitRef, ...}. Default = single format.")
    public String tsfileFormat = "Single";

    @Option(name = "-L", usage = "Length of default timestamp array. Default = 8")
    public int L = 8;
    
    @Option(name = "-permute", usage = "What algorithm to permute and get id list. Value = {A1H1, A1H2, A2}. A1 is non-random (Algorithm 1 and 2 in the paper), A2 is random with prior (Algorithm 3 in the paper). H1 and H2 are variants of A1. A, B, C, D are equivalent symmetry, e.g, {A1H1A, A1H1B, A1H1C, A1H1D; A1H2A, A1H2B, A1H2C, A1H2D; A2A, A2B, A2C, A2D}. Default = empty string, meaning simple random.")
    public String permuteAlgorithm = "";
}