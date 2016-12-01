/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.DataModel;

import cgs_lda_multicore.Utility.GeneralUtility;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;
import jgibblda.Dictionary;
import jgibblda.Document;
import jgibblda.LDADataset;
import static jgibblda.LDADataset.readDataSet;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

/**
 * Note: after divide train and test data, reading from train and test data file
 * would create 2 different vocabulary. -> need to implement a new reading
 * method for LDADataset to handle this problem.
 *
 * @author THNghiep
 */
public class DataPreparation {

    /**
     * Randomly divide word instances in each doc into training set and test
     * set. Save divided training set and test set to file.
     *
     * @param data
     * @param testSetProportion [0, 1]
     * @param dir
     * @param dfile
     * @return
     * @throws Exception
     */
    public static LDADataset prepareTestSet(LDADataset data, double testSetProportion, String dir, String dfile) throws Exception {
        // "data" have all doc index, word index, full doc content (via dictionary).
        // Full wordmap has been saved in PModel.readDataset(), right after reading datafile.
        // Whenever needed, call saving method of dictionary.

        // Now, randomly separate word instance.
        LDADataset dataTest = new LDADataset();
        dataTest.M = data.M;
        dataTest.V = data.V;
        dataTest.localDict = data.localDict;
        dataTest.docs = new Document[dataTest.M];
        for (int i = 0; i < data.M; i++) {
            // Separate each doc content.
            int sepIdx = (int) (data.docs[i].length * (1 - testSetProportion));
            // Reuse Document object in data.docs.
            int[] wordsDocsTrain = new int[sepIdx];
            // New Document object for dataTest.
            dataTest.docs[i] = new Document(data.docs[i].length - sepIdx);
            GeneralUtility.shufflePrimitiveArray(data.docs[i].words);
            for (int j = 0; j < data.docs[i].length; j++) {
                if (j < sepIdx) {
                    wordsDocsTrain[j] = data.docs[i].words[j];
                } else {
                    dataTest.docs[i].words[j - sepIdx] = data.docs[i].words[j];
                }
            }
            data.docs[i].words = wordsDocsTrain;
            data.docs[i].length = data.docs[i].words.length;
//            data.docs[i].rawStr = "";
        }
        // Finally, save train and test data.
        writeDataset(data, dir, "TrainData_" + dfile);
        writeDataset(dataTest, dir, "TestData_" + dfile);

        return dataTest;
    }

    /**
     * Simply write the dataset from LDADataset to data file in this lib format.
     * 
     * @param data
     * @param dir
     * @param dfile
     * @return
     * @throws Exception 
     */
    public static boolean writeDataset(LDADataset data, String dir, String dfile) throws Exception {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(dir + File.separator + dfile));
            writer.write(String.valueOf(data.M));
            writer.write("\n");
            for (int i = 0; i < data.M; i++) {
                for (int j = 0; j < data.docs[i].length; j++) {
                    writer.write(data.localDict.getWord(data.docs[i].words[j]) + " ");
                }
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Error while writing dataset: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Read dataset from datafile with a known full word map.
     * Format of datafile: 
     * First line: number of documents.
     * Each other line: each word separated by space.
     *
     * @param fileName
     * @param wordMap
     * @return
     */
    public static LDADataset readDataset(String fileName, String wordMap) throws Exception {
        try {
            // Read dictionary.
            Dictionary dict = new Dictionary();
            if (wordMap != null) {
                dict.readWordMap(wordMap);
            }

            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));

            // Read number of document.
            String line;
            line = reader.readLine();
            int M = Integer.parseInt(line);

            LDADataset data = new LDADataset(M);

            data.localDict = dict;

            // Read each document.
            for (int i = 0; i < M; ++i) {
                line = reader.readLine();
                // Add a doc parsed from string, based on localDict.
                data.setDoc(line, i);
                // Remove raw string to reduce space.
//                data.docs[i].rawStr = "";
            }

            reader.close();

            return data;
        } catch (Exception e) {
            System.out.println("Read Dataset Error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Read dataset in NYT format.
     * Ref: https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/readme.txt
     * 
     * @param fileName
     * @param wordMap
     * @return LDADataset.
     * @throws Exception 
     */
    public static LDADataset readDatasetNYT(String fileName, String wordMap) throws Exception {
        try {
            // Read document file.
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

            // Read number of documents.
            String line;
            line = reader.readLine();
            int M = Integer.parseInt(line);

            LDADataset data = new LDADataset(M);

            // Read number of words in vocabulary.
            line = reader.readLine();
            int V = Integer.parseInt(line);
            
            data.V = V;

            // Read number of doc-word pairs or number of remaining lines.
            line = reader.readLine();
            int NNZ = Integer.parseInt(line);

            // Read each doc-word pair.
            // Need to adjust docID and wordID to 0 based index, that is minus 1.
            int oldDocID = -1;
            ArrayList wordList = null;
            for (int i = 0; i < NNZ; ++i) {
                line = reader.readLine();
                String[] values = line.split(" ");
                if (values.length != 3) {
                    continue;
                }
                // Adjust ID.
                int docID = Integer.parseInt(values[0].trim()) - 1;
                int wordID = Integer.parseInt(values[1].trim()) - 1;
                int count = Integer.parseInt(values[2].trim());
                
                // If encounter new docID then add previous word list to data and reset word list.
                if (docID != oldDocID) {
                    if (wordList != null) {
                        data.docs[oldDocID] = new Document(wordList);
                    }
                    oldDocID = docID;
                    wordList = new ArrayList<>();
                }
                // Add new word instances to word list.
                for (int j = 0; j < count; j++) {
                    wordList.add(wordID);
                }
            }
            // Add the last document, where docID is max.
            if ((oldDocID != -1) && (wordList != null)) {
                data.docs[oldDocID] = new Document(wordList);
            }
            reader.close();
            
            // Debug null pointer.
            // There are some documents in NYT that have no word, they are null, now change them to empty.
            for (int i = 0; i < M; i++) {
                if (data.docs[i] == null) {
                    data.docs[i] = new Document(new ArrayList<Integer>());
                }
            }

            // Read dictionary.
            // Not actually needed for dataset, just for back reference when printing topic dist, may read separately later.
            Dictionary dict = null;
            if (wordMap != null) {
                dict = readWordMapNYT(wordMap, V);
            }
            data.localDict = dict;
            
            return data;
        } catch (Exception e) {
            System.out.println("Read Dataset Error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Read wordmap in NYT format.
     * Need to specify correct V: total number of word, this is read from docword file.
     * Index starts from 0 for compatibility with current code, instead of 1.
     * 
     * @param wordMapFile
     * @param V
     * @return 
     */
    public static Dictionary readWordMapNYT(String wordMapFile, int V) {
        try {
            Map<String, Integer> word2id;
            Map<Integer, String> id2word;
            word2id = new HashMap<>(2 * V, 1f);
            id2word = new HashMap<>(2 * V, 1f);
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(wordMapFile), "UTF-8"));
            // read wordmap.
            // Note: 
            // - only read to V, so V must be correct.
            // - start index from 0, instead of 1 (NYT default).
            for (int wordID = 0; wordID < V; ++wordID) {
                String word = reader.readLine();
                if (word.isEmpty()) {
                    continue;
                }
                id2word.put(wordID, word);
                word2id.put(word, wordID);
            }
            reader.close();
            
            Dictionary dict = new Dictionary(word2id, id2word);
            return dict;
        } catch (Exception e) {
            System.out.println("Error while reading wordmap:" + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    public static LDADataset readDatasetCORE(String fileName) throws Exception {
        try {
            // Read document file.
            BufferedReader reader = null;
            
            if (fileName.endsWith(".tar.gz")) {
                // This case read from .tar.gz file.
                TarArchiveInputStream tAIS = new TarArchiveInputStream(
                                                    new GZIPInputStream(
                                                    new FileInputStream(fileName)));
                TarArchiveEntry tarArchiveEntry;
                
                while ((tarArchiveEntry = tAIS.getNextTarEntry()) != null) {
                  if (tarArchiveEntry.isFile()) {
                      reader = new BufferedReader(
                                    new InputStreamReader(
                                    new FileInputStream(tarArchiveEntry.getFile()), "UTF-8"));
                      String line;
                      
                      while ((line = reader.readLine()) != null) {
                          // Process line, each line is a json of a document.
                      }
                      reader.close();
                  }
                }
                tAIS.close();
            }
            return null;
        } catch (Exception e) {
            System.out.println("Read Dataset Error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}
