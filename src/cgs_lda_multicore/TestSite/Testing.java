/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.TestSite;

import cgs_lda_multicore.UI.PLDA;

/**
 *
 * @author THNghiep
 */
public class Testing {
    public static void main(String args[]) {
        if (args.length == 0) {
            // Example parameter:
            // Estimate: -est -alpha 0.5 -beta 0.1 -ntopics 100 -niters 1000 -savestep 100 -twords 20 -dfile models/casestudy/newdocs.dat
            // Continue estimate: -estc -dir models/casestudy/ -model model-01000 -niters 800 -savestep 100 -twords 30
            // Inference: -inf -dir models/casestudy/ -model model-01800 -niters 30 -twords 20 -dfile newdocs.dat
            args = new String[]{"-est", 
                "-dir", "D:\\ResD\\temp\\CGS_LDA_Multicore_Test", 
                "-dfile", "docword.nips.txt", "-wordmap", "vocab.nips.txt",
                "-testsetprop", "0.1",
                "-datafileformat", "NYT",
                "-dfiletrain", "",
                "-dfiletest", "",
                "-alpha", "0.5", "-beta", "0.1", 
                "-ntopics", "150", 
                "-niters", "1", "-burnin", "50", "-savestep", "0",
                "-twords", "100", 
                "-howtogetdist", "1",
                "-threadpoolsize", "1",
                "-P", "1", "-shuffle", "1", "-howtopart", "2"};
            
            PLDA.main(args);
        }
    }
}
