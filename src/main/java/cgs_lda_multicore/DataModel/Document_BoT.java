/*
 * Copyright (C) 2007 by
 * 
 * 	Xuan-Hieu Phan
 *	hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
 * 	Graduate School of Information Sciences
 * 	Tohoku University
 * 
 *  Cam-Tu Nguyen
 *  ncamtu@gmail.com
 *  College of Technology
 *  Vietnam National University, Hanoi
 *
 * JGibbsLDA is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * JGibbsLDA is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JGibbsLDA; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
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
     * Create new document_bot from document.
     * Reuse data directly.
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
