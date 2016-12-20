/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.DataModel;

import jgibblda.*;
import java.util.ArrayList;
import java.util.Vector;

/**
 * Add method with ArrayList param.
 *
 * Remove raw string rawStr completely to optimize size.
 *
 * @modify THNghiep
 */
public class Document_BoT extends Document {

    //----------------------------------------------------
    //Instance Variables
    //----------------------------------------------------
    /**
     * Timestamp array.
     */
    public int[] tss;
    /**
     * Length of timestamp array.
     */
    public int L;

    //----------------------------------------------------
    //Constructors
    //----------------------------------------------------
    public Document_BoT() {
        super();
        tss = null;
        L = 0;
    }

    public Document_BoT(int length, int lengthts) {
        super(length);
        this.L = lengthts;
        tss = new int[lengthts];
    }

    public Document_BoT(int length, int[] words, int lengthts, int[] tss) {
        super(length, words);
        this.L = length;
        this.tss = new int[lengthts];
        for (int i = 0; i < lengthts; ++i) {
            this.tss[i] = tss[i];
        }
    }

    public Document_BoT(ArrayList<Integer> doc, ArrayList<Integer> tsList) {
        super(doc);
        this.L = tsList.size();
        this.tss = new int[L];
        for (int i = 0; i < L; i++) {
            this.tss[i] = tsList.get(i);
        }
    }

    /**
     * Create new document_bot from document. Reuse data directly.
     *
     * @param doc
     */
    public Document_BoT(Document doc) {
        super();
        this.length = doc.length;
        this.words = doc.words;
        this.L = 0;
        this.tss = null;
    }
}
