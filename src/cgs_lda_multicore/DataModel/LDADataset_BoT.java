/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.DataModel;

import static cgs_lda_multicore.DataModel.DataPreparation.readWordMapNYT;
import jgibblda.*;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.zip.GZIPInputStream;

/**
 * This class contain the source data of all documents in the corpus: Document
 * and word-id mapping.<br/>
 * Document has raw string and word id array corresponding to mapping in this
 * class. Read dataset from file.
 * 
 * Modify: change Vector to ArrayList.
 *
 * @documentation THNghiep
 */
public class LDADataset_BoT extends LDADataset {
    /**
     * Dictionary of timestamp.
     */
    public Dictionary localDictBoT;
    /**
     * a list of documents.
     */
    public Document_BoT[] docs;

    /**
     * Number of unique timestamp tokens.
     */
    public int O;

    //--------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------
    public LDADataset_BoT() {
        super();
        localDictBoT = new Dictionary();
        docs = null;
        O = 0;
    }

    public LDADataset_BoT(int M) {
        super(M);
        localDictBoT = new Dictionary();
        docs = new Document_BoT[M];
        this.O = 0;
    }

    public LDADataset_BoT(int M, Dictionary globalDict) {
        super(M, globalDict);
        localDictBoT = new Dictionary();
        docs = new Document_BoT[M];
        this.O = 0;
    }
    
    public LDADataset_BoT(LDADataset data) {
        super();
        
        // Transfer data.
        this.M = data.M;
        this.V = data.V;
        this.localDict = data.localDict;
        
        localDictBoT = new Dictionary();
        this.docs = new Document_BoT[M];
        for (int i = 0; i < M; i++) {
            this.docs[i] = new Document_BoT(data.docs[i]);
        }
        super.docs = this.docs;
        
        // Size of timestamp vocabulary, not yet read TS data so set to 0.
        this.O = 0;
    }

    /**
     * Read the very simple TS data: only 1 timestamp.
     * Format: 1 file.
     * [O]
     * [First TS]
     * [Last TS]
     * [TS of doc 1]
     * [TS of doc 2]
     * ...
     * In MAS data, O is the length of the range from the first to the last publication.
     * 
     * @param data
     * @param fileName
     * @param L
     * @throws Exception 
     */
    static void readTSDataMASSingle(LDADataset_BoT data, String fileName, int L) throws Exception {
        try {
            // Read document ts file.
            BufferedReader reader = null;
            if (fileName.endsWith(".txt")) {
                // This case read from txt file.
                reader = new BufferedReader(
                                new InputStreamReader(
                                new FileInputStream(fileName), "UTF-8"));
            } else if (fileName.endsWith(".gz")) {
                // This case read from .gz file.
                reader = new BufferedReader(
                                new InputStreamReader(
                                new GZIPInputStream(
                                new FileInputStream(fileName)), "UTF-8"));
            }

            // Read number of TS, set O.
            String line;
            line = reader.readLine();
            data.O = Integer.parseInt(line);

            line = reader.readLine();
            int firstTS = Integer.parseInt(line);

            line = reader.readLine();
            int lastTS = Integer.parseInt(line);

            // Read TS data.
            for (int m = 0; m < data.M; m++) {
                // Read TS of each doc.
                line = reader.readLine();
                if (line.trim().isEmpty()) {
                    continue;
                }
                int ts = Integer.parseInt(line.trim());
                // Add to dictionary or get ID.
                int tsID = data.localDictBoT.addWord(String.valueOf(ts));
                // Create TSs data in doc.
                data.docs[m].tss = new int[L];
                // Divide ts array.
                int startMain = Math.round((float) L / 4);// 0:0 1:0 2:1 3:1 4:1 5:1 6:2 7:2 8:2
                int endMain = Math.round((float) L * 3 / 4);// 0:0 1:1 2:2 3:2 4:3 5:4 6:5 7:5 8:6
                // Fill in ts array.
                for (int i = 0; i < L; i++) {
                    if (i < startMain && ts != firstTS) {
                        data.docs[m].tss[i] = data.localDictBoT.addWord(String.valueOf(ts - 1));
                    }
                    if (i >= startMain && i < endMain) {
                        data.docs[m].tss[i] = tsID;
                    }
                    if (i >= endMain && ts != lastTS) {
                        data.docs[m].tss[i] = data.localDictBoT.addWord(String.valueOf(ts + 1));
                    }
                }
                if (ts == firstTS) {
                    data.docs[m].tss = Arrays.copyOfRange(data.docs[m].tss, startMain, L);
                } else if (ts == lastTS) {
                    data.docs[m].tss = Arrays.copyOfRange(data.docs[m].tss, 0, endMain);
                }
                data.docs[m].L = data.docs[m].tss.length;
            }

            reader.close();
        } catch (Exception e) {
            System.out.println("Read Dataset Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    static void readTSDataArray(LDADataset_BoT data, String string) throws Exception {
        
    }

    static void readTSDataCitRef(LDADataset_BoT data, String string) throws Exception {
        
    }

    /**
     * Print dataset statistics.
     * 
     * @throws Exception 
     */
    public void printDatasetStatistics() throws Exception {
        System.out.println("Number of documents M: " + M);
        System.out.println("Number of unique words V: " + V);
        System.out.println("Number of unique timestamps O: " + O);
        long wordInstance = 0;
        long tsInstance = 0;
        for (Document_BoT doc : docs) {
            wordInstance += doc.length;
            tsInstance += doc.L;
        }
        System.out.println("Number of word instances: " + wordInstance);
        System.out.println("Number of timestamp instances: " + tsInstance);
    }

    /**
     * Print information for unit test.
     * Output:
     * - Paper after every stepSize: 
     * -> title, abstract.
     * -> timestamp array.
     * 
     * @param stepSize
     * @throws Exception 
     */
    public void printDatasetUnitTestInfo(int stepSize) throws Exception {
        for (int i = 0; i < docs.length/stepSize; i++) {
            System.out.print(String.valueOf(i * stepSize) + ".");
            System.out.print("\nContent: ");
            for (int w : docs[i * stepSize].words) {
                System.out.print(localDict.getWord(w) + " ");
            }
            System.out.print("\nTimestamp: ");
            for (int t : docs[i * stepSize].tss) {
                System.out.print(localDictBoT.getWord(t) + " ");
            }
            System.out.print("\n");
        }
    }
}
