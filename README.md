## Parallel_Topic_Modeling: An Efficient Time-aware Topic Modeling Parallelization
### Introduction
Parallel_Topic_Modeling is a Java implementation of parallel Gibbs sampling for topic modeling based on a novel <a href="https://arxiv.org/pdf/1510.04317.pdf">data partitioning algorithm</a>. The source code supports learning topic distributions of large corpus using <a href="http://www.jmlr.org/papers/v3/blei03a.html">LDA</a> and <a href="http://link.springer.com/chapter/10.1007/978-3-642-00672-2_51">BoT</a> (a simple and effective time-aware topic model), with efficient parallelization speedup on multi-core CPU. Our code is based on <a href="http://jgibblda.sourceforge.net/">JGibbLDA</a>, a non-parallel LDA implementation.

### Data
We also built a dataset that contains time-stamped documents for time-aware topic modeling experiments. The dataset contains over 1 million scientific papers in the computer science domain from 1951 to 2010, crawled from <a href="http://academic.research.microsoft.com">Microsoft Academic Search</a> (MAS), finished in November 2012. The dataset can be downloaded at: https://drive.google.com/file/d/0B8gXe63FdGk5OEVfaVl6VTlsb2c (.zip, 595 MB).

#### Data format:
_Document file:_
```
Line 1: number of documents.
Line 2: word list of document 1, each word separated by space.
Line 3: word list of document 2, each word separated by space.
...
```

_Timestamp file:_
```
Line 1: number of distinct timestamps
Line 2: earliest timestamp
Line 3: latest timestamp
Line 4: timestamp of document 1
Line 5: timestamp of document 2
...
```

The source code also supports other popular dataset formats, such as <a href="http://archive.ics.uci.edu/ml/datasets/Bag+of+Words">NIPS and NYTimes</a>.

### How to use
Parallel_Topic_Modeling can either be used in command line or as a library in you code. You can specify the model (LDA or BoT), the number of parallel threads, how to partition data, etc. You can use the <a href="https://github.com/tranhungnghiep/Parallel_Topic_Modeling/releases">pre-built jar file</a> directly.

#### Example:
_LDA Model:_
</br>The following command estimates the topic distribution of MAS corpus (the above dataset) using LDA model, parallelized in 10 threads, where MAS_doc_removedSW.txt is document file.
</br>`java -Xmx64g -cp ./Code/Parallel_Topic_Modeling.jar cgs_lda_multicore.UI.PLDA -est -dir ./Data/MAS -dfile MAS_doc_removedSW.txt -datafileformat Private -dfiletrain -dfiletest -alpha 0.5 -beta 0.1 -ntopics 150 -niters 200 -burnin 100 -savestep 0 -twords 100 -howtogetdist 1 -threadpoolsize 0 -P 10 -shuffle 10 -howtopart 2`

It can be shortened using default value as:
</br>`java -Xmx64g -cp ./Code/Parallel_Topic_Modeling.jar cgs_lda_multicore.UI.PLDA -est -dir ./Data/MAS -dfile MAS_doc_removedSW.txt -alpha 0.5 -beta 0.1 -threadpoolsize 0 -P 10`

_BoT Model:_
</br>The following command estimates the topic distribution of MAS corpus (the above dataset) using BoT model, parallelized in 10 threads, where MAS_doc_removedSW.txt is document file and MAS_ts.txt is timestamp file. Note that for BoT model, we change the class to cgs_lda_multicore.UI.PLDA_BoT. We also specify some new options such as timestamp file, timestamp file format, time prior gamma, time array length L, how to shuffle timestamps.
</br>`java -Xmx64g -cp ./Code/Parallel_Topic_Modeling.jar cgs_lda_multicore.UI.PLDA_BoT -est -dir ./Data/MAS -dfile MAS_doc_removedSW.txt -tsfile MAS_ts.txt -datafileformat Private -tsfileformat Single -dfiletrain -dfiletest -alpha 0.5 -beta 0.1 -gamma 0.1 -ntopics 150 -L 8 -niters 200 -burnin 100 -savestep 0 -twords 100 -howtogetdist 1 -threadpoolsize 0 -P 10 -shuffle 10 -shufflets 100 -howtopart 2`

It can be shortened using default value as:
</br>`java -Xmx64g -cp ./Code/Parallel_Topic_Modeling.jar cgs_lda_multicore.UI.PLDA_BoT -est -dir ./Data/MAS -dfile MAS_doc_removedSW.txt -tsfile MAS_ts.txt -alpha 0.5 -beta 0.1 -gamma 0.1 -threadpoolsize 0 -P 10`

Please see files `/src/jgibblda/LDACmdOption.java` and `/src/cgs_lda_multicore/Utility/PLDACmdOption.java` for explanation of commandline options.

For using in your code as a library, please see sample files `/src/cgs_lda_multicore/TestSite/Testing.java`	for LDA and `/src/cgs_lda_multicore/TestSite/Testing_BoT.java` for BoT.

The source code was written in Java using Ant-based NetBeans project format, which can be opened in <a href="https://netbeans.org/">NetBeans IDE</a>. For other IDEs, please see the <a href="https://github.com/tranhungnghiep/Parallel_Topic_Modeling/tree/Parallel_Topic_Modeling_Maven">Maven port</a>. 

### License
Parallel_Topic_Modeling is a free software under <a href="http://www.gnu.org/licenses/gpl.html">GNU GPL</a> 3.0.

The dataset is provided under open <a href="http://opendatacommons.org/licenses/by/summary/">ODC-BY License</a> 1.0.

**Corresponding paper:**  
If you find the codes or data useful, please cite the following paper.  
*Hung Nghiep Tran, Atsuhiro Takasu. <a href="http://ieeexplore.ieee.org/document/7334854/" target="_blank">Partitioning Algorithms for Improving Efficiency of Topic Modeling Parallelization</a>. PacRim 2015.*

For more information, please visit the website: https://sites.google.com/site/tranhungnghiep
