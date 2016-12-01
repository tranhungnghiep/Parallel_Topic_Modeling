/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cgs_lda_multicore.Utility;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * @author THNghiep
 */
public class GeneralUtility {
    public static int getNumOfCore() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.availableProcessors();
    }

    public static void shufflePrimitiveArray(int[] array) throws Exception {
      List<Integer> list = new ArrayList<>();
      for (int i : array) {
        list.add(i);
      }

      Collections.shuffle(list);

      for (int i = 0; i < list.size(); i++) {
        array[i] = list.get(i);
      }    
    }

    public static LinkedHashMap<Integer, Long> arrayToLinkedHashMap(AtomicLong[] array) throws Exception {
        int length = array.length;        
        LinkedHashMap<Integer, Long> hm = new LinkedHashMap<>(2 * length, 1f);
        
        for (int i = 0; i < length; i++) {
            hm.put(i, array[i].get());
        }
        
        return hm;
    }

    /**
     * getSortedMapDescending
     * @param map
     * @return
     */
    public static LinkedHashMap getSortedMapDescending(HashMap map) throws Exception {
        if (map == null) {
            return null;
        }
        List list = new LinkedList(map.entrySet());
        Collections.sort(list, new Comparator() {
            @Override
            public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o2)).getValue())
                    .compareTo(((Map.Entry) (o1)).getValue());
            }
        });

        LinkedHashMap result = new LinkedHashMap();
        for (Iterator it = list.iterator(); it.hasNext();) {
            Map.Entry entry = (Map.Entry) it.next();
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    /**
     * Copy from 1 list to another list, interposing head/tail.
     * 
     * @param list
     * @return
     * @throws Exception 
     */
    public static List interposeList(List list) throws Exception {
        int length = list.size();
        List newList = new ArrayList(length);
        
        for (int i = 0; i < length; i++) {
            if (i % 2 == 0) {
                newList.add(list.get(i / 2));
            } else {
                newList.add(list.get(length - 1 - i / 2));
            }
        }
        
        return newList;
    }

    public static List interposeSymmetryList(List list) throws Exception {
        int length = list.size();

        for (int i = 0; i < length / 2; i++) {
            if (i % 2 == 1) {
                Object temp = list.get(i);
                list.set(i, list.get(length - i));
                list.set(length - i, temp);
            }
        }
        
        return list;
    }

    /**
     * Given a sorted descending list, cut to P part, each part contains sample from all range of list.
     * These samples are randomly got from each range.
     * Finally randomly shuffle each part.
     * 
     * @param list
     * @param P
     * @return
     * @throws Exception 
     */
    public static List rangerList(List list, int P) throws Exception {
        int length = list.size();
        List newList = new ArrayList(length);
        
        final List<List<Object>> partList = new ArrayList<>(P);
        List<Integer> partIDList = new ArrayList(P);
        for (int i = 0; i < P; i++) {
            partList.add(new ArrayList<>());
            partIDList.add(i);
        }
        
        // Cut P part, each part contains samples from all ranges.
        for (int i = 0; i < length; i++) {
            if (i % P == 0) {
                // Sample are randomly got from each range.
                Collections.shuffle(partIDList);
            }
            partList.get(partIDList.get(i % P)).add(list.get(i));
        }

        // Finally shuffle each part.
        // Set up thread pool.
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
        for (int i = 0; i < P; i++) {
            final int partID = i;
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        Collections.shuffle(partList.get(partID));
                    } catch (Exception ex) {
                        System.err.println(ex.toString());
                        ex.printStackTrace();
                    }
                }
            });
        }
        // Shutdown thread pool, wait until shutdown finished.
        executor.shutdown();
        while (!executor.isTerminated()) {}

        // Do not concurrent modify list size, instead, add sequentially.
        for (int i = 0; i < P; i++) {
            newList.addAll(partList.get(i));
        }
        
        return newList;
    }

    /**
     * Swap a list head to tail.
     * 
     * @param list
     * @throws Exception 
     */
    public static void swapList(List list) throws Exception {
        int length = list.size();
        for (int i = 0; i < length / 2; i++) {
            Object temp = list.get(i);
            list.set(i, list.get(length - 1 - i));
            list.set(length - i - 1, temp);
        }
    }
}
