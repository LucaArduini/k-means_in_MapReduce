package it.unipi.hadoop;

import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class KMeans{

    public static String logFile = "outputsLog.txt";
    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, ClusteringFeature> {
        // Hadoop Map Types (K1, V1) ---> (K2, L(V2))
        // K1 --> Object (Not useful)
        // V1 --> String representation of a Point
        // K2 --> Cluster Index in which the points have been assigned to
        // V2 --> Clustering Feature (partial sum of points, number of points)
        ArrayList<Point> centroids;
        ArrayList<ClusteringFeature> clusteringFeatureList;

        public void setup(Context context) throws IOException, InterruptedException {
            // initialization
            centroids = new ArrayList<>();
            clusteringFeatureList = new ArrayList<>();

            // cast centroid string to an array of points (centroids)
            String centroidsString = context.getConfiguration().get("centroids");
            String[] splits = centroidsString.split("\\n");
            for(int i= 0; i < splits.length; i++)
                centroids.add(parsePoint(splits[i]));

            for(int i=0; i<centroids.size(); i++){
                // Initializing clustering features < index_cluster , <[0....0], 0>. This is needed in order to sum later.
                clusteringFeatureList.add(new ClusteringFeature(centroids.get(0).getFeatures().size()));
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Point point = parsePoint(value.toString());
            // Qui decido a quale cluster assegnare il punto preso da input
            int indexNearestCentroid = point.nearestCentroid(centroids);
            // Aggiungi alla parte "somma parziale" della clustering feature di indice indexNearestCentroid il nuovo punto ricevuto da map()
            clusteringFeatureList.get(indexNearestCentroid).getPartialSum().sumPoint(point);
            // Segnalo che nella clustering feature di indice indexNearestCentroid c'è un nuovo punto
            clusteringFeatureList.get(indexNearestCentroid).setNumPoints(clusteringFeatureList.get(indexNearestCentroid).getNumPoints()+1);
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            for(int i=0; i<centroids.size(); i++)
                context.write(new IntWritable(i), clusteringFeatureList.get(i));
        }
    }

    public static class KMeansReducer extends Reducer<IntWritable, ClusteringFeature, IntWritable, Point>{
        // K2 = IntWritable (index cluster)
        // V2 = ClusteringFeature (coppia)
        // K3 = IntWritable (index cluster)
        // V3 = Point (centroide di quel cluster)

        public void reduce(IntWritable key, Iterable<ClusteringFeature> values, Context context) throws IOException, InterruptedException {
            ArrayList<ClusteringFeature> listCouples = new ArrayList<>();
            for(ClusteringFeature cf : values){
                ClusteringFeature toAdd = new ClusteringFeature(cf.getPartialSum(), cf.getNumPoints());
                listCouples.add(toAdd);
            }
            // result contains the sum of all the clustering features coming 
            ClusteringFeature result = new ClusteringFeature(listCouples);

            // compute the average and store it in a point
            Point centroid = result.computeMean();

            // send in the output
            context.write(key, centroid);           //nella forma: "0       <4.0019444906464745, 4.546128116278345>"
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (otherArgs.length != 5) {  // input, k, niter, output, dim
            System.err.println("Usage: KMeans <input> <k> <max_iter> <output>");
            System.exit(1);
        }
        System.out.println("args[0]: <input>=" + otherArgs[0]);
        System.out.println("args[1]: <k>=" + otherArgs[1]);
        System.out.println("args[2]: <max_iter>=" + otherArgs[2]);
        System.out.println("args[3]: <output>=" + otherArgs[3]);
        System.out.println("args[4]: <d>=" + otherArgs[4]);

        // initial random centroids computation
        int k = Integer.parseInt(otherArgs[1]);
        ArrayList<Point> initialCentroids = Point.getPoints(k, Integer.parseInt(otherArgs[4]));

        int iter = 0;
        int MAX_ITER = Integer.parseInt(otherArgs[2]);
        long start = System.currentTimeMillis();
        log("START");
        log("Dataset : " + otherArgs[0]);
        log("------------------------------------------------------------");
        while (iter < MAX_ITER) {
            Job job = Job.getInstance(conf, "ParallelKMeans");
            job.setJarByClass(KMeans.class);

            // set mapper/reducer
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);

            // define mapper's output key-value
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(ClusteringFeature.class);

            // define reducer's output key-value
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Point.class);

            // define I/O
            FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
            FileOutputFormat.setOutputPath(job, new Path(otherArgs[3]+iter));
            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            System.out.println("[ITER: " + iter + " ] centroids: "+initialCentroids.toString());

            // creating a String s that contains the centroid to send to the mapper
            StringBuilder s = new StringBuilder();
            for(int i = 0; i < k; i++){
                s.append(initialCentroids.get(i).toString() + '\n');
            }
            job.getConfiguration().setStrings("centroids", s.toString());

            //////////// JOB START /////////////
            job.waitForCompletion(true);
            ////////////////////////////////////

            // read the new centroids computed by the job. otherArgs[3] = name of the output file
            ArrayList<Point> newCentroids = new ArrayList<Point>();
            newCentroids = readAndAddCentroid(conf, new Path(otherArgs[3]+iter), k);

            iter++;

            // Logging
            // --------------------------------------------
            for(int i = 0; i < initialCentroids.size(); i++){
                log("Centroide all'iterazione precedente (" + (iter-1) + ") " + initialCentroids.get(i).toString());
                log("Nuovo Centroide iterazione (" + (iter) + ") " + newCentroids.get(i).toString());
            }
            // --------------------------------------------
            // check if the centroids have changed
            if (checkTermination(initialCentroids, newCentroids)) {
                System.out.println("[ITER: " + iter + " ] centroids: "+initialCentroids.toString());
                break;
            }
            


            initialCentroids = newCentroids;
        }
        long end = System.currentTimeMillis();
        long time = end - start;
        log("FINITO : ci sono volute " + (iter) + " iterazioni e " + time + " millisecondi");
        log("------------------------------------------------------------");
    }

    private static void log(String msg) throws IOException{
        FileWriter fileWriter = new FileWriter(logFile, true);
        fileWriter.write(msg + '\n');
        fileWriter.close();
    }

    private static boolean checkTermination(ArrayList<Point> initialCentroids, ArrayList<Point> newCentroids) {
        for(int i = 0; i < initialCentroids.size(); i++){
            if(initialCentroids.get(i).equals(newCentroids.get(i))==false)
                return false;
        }
        return true;
    }

    // FUNZIONE DI TERMINAZIONE QUALORA PULIAFITO DICESSE CHE VA BENE
    /* --------------------------------------------------------------------------
    private static boolean checkTermination2(ArrayList<Point> initialCentroids, ArrayList<Point> newCentroids, double epsilon) {
        int sum = 0;
        for(int i = 0; i < initialCentroids.size(); i++){
            sum += initialCentroids.get(i).distance(newCentroids.get(i));
        }
        if(sum < epsilon)
            return true;
        return false;
    }
    */



    private static ArrayList<Point> readAndAddCentroid(Configuration conf, Path outputPath, int k) throws IOException {
        // Function used to read centroids computed by the job and sent in the output file in the HDFS
        // It reads them and returns an ArrayList<Point>

        FileSystem fs = FileSystem.get(conf);
        FileStatus[] fileStatuses = fs.listStatus(outputPath);      //ex: outputPath = /user/hadoop/output_angelo0
        ArrayList<Point> centroidsList = new ArrayList<>(k);
        //boolean output_found = false;

        boolean[] test = new boolean[k];    //inizializzato a false di default

        for (FileStatus status : fileStatuses) {
            if (!status.isDirectory()) {
                Path filePath = status.getPath();
                if (filePath.getName().startsWith("part-r-")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(filePath)));
                    String line;
                    while ((line = br.readLine()) != null) {
                        System.out.println("linea letta dall'output del reducer: "+line);
                        //centroidsList.add(parsePoint(line.substring(2)));

                        int read_centroids = Integer.parseInt(line);
                        centroidsList.set(read_centroids, parsePoint(line.substring(2)));
                        test[read_centroids] = true;
                    }
                    br.close();
                    //output_found = true;
                }
            }
        }
        //return (output_found)? centroidsList : null;
        for(boolean x : test){
            if(x==false){
                System.err.println("It was not possible to read all the centroids");
                System.exit(1);
            }
        }
        return centroidsList;
    }

    private static Point parsePoint(String str) {
        // Takes a string in input: <0.41410840, 1.48714702> and returns a Point
        String cleanInput = str.replaceAll("[<>]", "");   //rimpiazza un singolo carattere che è '<' o '>'

        String[] numbersArray = cleanInput.split(",\\s*");          //una virgola seguita da zero o più spazi bianchi
        ArrayList<Double> numbersList = new ArrayList<>();

        for (String numberStr : numbersArray) {
            double number = Double.parseDouble(numberStr);
            numbersList.add(number);
        }

        return new Point(numbersList);
    }
}
