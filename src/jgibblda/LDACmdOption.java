package jgibblda;

import org.kohsuke.args4j.*;

public class LDACmdOption {

    @Option(name = "-est", usage = "Specify whether we want to estimate model from scratch")
    public boolean est = false;

    @Option(name = "-estc", usage = "Specify whether we want to continue the last estimation")
    public boolean estc = false;

    @Option(name = "-inf", usage = "Specify whether we want to do inference")
    public boolean inf = false;

    @Option(name = "-dir", usage = "Specify directory")
    public String dir = "";

    @Option(name = "-dfile", usage = "Specify data file")
    public String dfile = "";

    @Option(name = "-model", usage = "Specify the model name")
    public String modelName = "";

    @Option(name = "-alpha", usage = "Specify alpha")
    public double alpha = -1.0;

    @Option(name = "-beta", usage = "Specify beta")
    public double beta = -1.0;

    @Option(name = "-ntopics", usage = "Specify the number of topics")
    public int K = 150;

    @Option(name = "-niters", usage = "Specify the number of iterations. Default = 200.")
    public int niters = 200;

    @Option(name = "-savestep", usage = "Thinning interval: Specify the number of steps to save the model since the last save. Save step = 0 means not save. Default = 0.")
    public int savestep = 0;

    @Option(name = "-twords", usage = "Specify the number of most likely words to be printed for each topic")
    public int twords = 100;

    @Option(name = "-withrawdata", usage = "Specify whether we include raw data in the input")
    public boolean withrawdata = false;

    @Option(name = "-wordmap", usage = "Specify the wordmap filename, used for input.")
    public String wordMapFileName = "";
}
