/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.TestSite;

import cgs_lda_multicore.UI.PLDA;
import cgs_lda_multicore.UI.PLDA_BoT;

/**
 *
 * @author THNghiep
 */
public class Testing_BoT {
    public static void main(String args[]) {
        if (args.length == 0) {
            // Example parameter:
            // Estimate: -est -alpha 0.5 -beta 0.1 -ntopics 100 -niters 1000 -savestep 100 -twords 20 -dfile models/casestudy/newdocs.dat
            // Continue estimate: -estc -dir models/casestudy/ -model model-01000 -niters 800 -savestep 100 -twords 30
            // Inference: -inf -dir models/casestudy/ -model model-01800 -niters 30 -twords 20 -dfile newdocs.dat
            args = new String[]{"-est", 
                "-dir", "D:\\ResD\\temp\\CGS_LDA_Multicore_Test\\LocalTestBoT", 
                "-dfile", "doc.txt", "-tsfile", "ts.txt",
                // "-isseptestset": absent means false.
                "-testsetprop", "0.1",
                "-datafileformat", "Private",
                "-tsfileformat", "Single",
                "-dfiletrain", "",
                "-dfiletest", "",
                "-alpha", "0.5", "-beta", "0.1", "-gamma", "0.1",
                "-ntopics", "150", "-L", "1",
                "-niters", "2", "-burnin", "50", "-savestep", "0",
                "-twords", "10", 
                "-howtogetdist", "1",
                "-threadpoolsize", "1",
                "-P", "10", "-shuffle", "1000", "-howtopart", "2"
            };
            
            PLDA_BoT.main(args);
        }
    }
}
