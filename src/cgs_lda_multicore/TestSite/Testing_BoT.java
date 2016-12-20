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
//                "-dir", "/Users/mac/NetBeansProjects/Parallel_Topic_Modeling/TestData", 
//                "-dfile", "doc.txt", "-tsfile", "ts.txt",
                "-dir", "E:\\NghiepTH Working\\Data\\PTM\\Test", 
                "-dfile", "MAS_doc_cleanToken_lowercase_removedSW_lemma.txt", "-tsfile", "MAS_ts.txt",
//                 //"-isseptestset": absent means false.
//                "-testsetprop", "0.1",
//                "-datafileformat", "Private",
//                "-tsfileformat", "Single",
//                "-dfiletrain", "",
//                "-dfiletest", "",
                "-alpha", "0.5", "-beta", "0.1", "-gamma", "0.1",
//                "-ntopics", "150", "-L", "8",
//                "-niters", "200", "-burnin", "100", "-savestep", "0",
//                "-twords", "100", 
//                "-howtogetdist", "1",
                "-threadpoolsize", "0",
//                "-P", "10", "-shuffle", "10", "-shufflets", "100", "-howtopart", "2"
                "-P", "1", "-shuffle", "1", "-shufflets", "1", "-howtopart", "2"
            };
            
            PLDA_BoT.main(args);
        }
    }
}
