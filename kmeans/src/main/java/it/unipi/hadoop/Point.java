package it.unipi.hadoop;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

public class Point implements Writable{
    private ArrayList<Double> features = new ArrayList<>();
    private int dim;

    public Point(int d){
        dim = d;
        for(int i = 0; i < d; i++){
            features.add(0.0);
        }
    }
    
    public Point(){
        
    }
    
    public Point(ArrayList<Double> features) {
        dim = features.size();
        this.features = features;
    }

    public ArrayList<Double> getFeatures() {
        return features;
    }

    public int getDim(){
        return dim;
    }

    public void setFeatures(ArrayList<Double> features) {
        this.features = features;
    }

    public void sumPoint(Point p){
        for(int i = 0; i < p.getFeatures().size(); i++){
            features.set(i, features.get(i) + p.getFeatures().get(i));
        }
    }

    private Double distance(Point p){
        Double sum = 0.0;

        for(int i = 0; i < features.size(); i++){
            sum += pow(features.get(i) - p.getFeatures().get(i), 2);
        }

        sum = sqrt(sum);
        return sum;
    }

    public int nearestCentroid(ArrayList<Point> centroids){
        // INIT
        int index_min = 0;
        double dist_min = distance(centroids.get(0));

        // SCORRI
        for(int i = 1; i < centroids.size(); i++){
            double d = distance(centroids.get(i));
            if(d < dist_min){
                dist_min = d;
                index_min = i;
            }
        }
        return index_min;
    }

    public void scale(int n){
        for(int i = 0; i < features.size(); i++){
            features.set(i, features.get(i) / n);
        }
    }

    public String toString() {
        StringBuilder str=new StringBuilder("");
        for (int i = 0; i < features.size(); i++) {
            str.append(features.get(i));

            if (i < features.size() - 1) {
                str.append(", ");
            }
        }
        return "<" + str + ">";
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeInt(dim);
        for(int i = 0; i < dim; i++)
            dataOutput.writeDouble(features.get(i));
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        dim = dataInput.readInt();
        for(int i = 0; i < dim; i++)
            features.add(dataInput.readDouble());
    }

    public boolean equals(Point p){
        for(int i = 0; i < dim; i++){
            if((double) features.get(i) != (double) p.getFeatures().get(i)){
                return false;
            }
        }
        return true;
    }
}

