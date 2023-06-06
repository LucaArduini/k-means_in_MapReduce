package it.unipi.hadoop;

import java.io.*;
import java.util.ArrayList;
import org.apache.hadoop.io.Writable;

public class ClusteringFeature implements Writable{
    private Point partialSum;
    private int numPoints = 0;

    public void write(DataOutput out) throws IOException{
        partialSum.write(out);
        out.writeInt(numPoints);
    }

    public void readFields(DataInput in) throws IOException{
        partialSum = new Point();
        partialSum.readFields(in);
        numPoints = in.readInt();
    }
    
    public ClusteringFeature(){
        //nothing
    }

    public ClusteringFeature(Point partialSum, int numPoints) {
        this.partialSum = partialSum;
        this.numPoints = numPoints;
    }

    public ClusteringFeature(int dim) {
        partialSum = new Point(dim);
    }

    public ClusteringFeature(ArrayList<ClusteringFeature> list){
        this.partialSum = list.get(0).getPartialSum();
        this.numPoints = list.get(0).getNumPoints();
        for (int i = 1; i < list.size(); i++) {
            this.partialSum.sumPoint(list.get(i).getPartialSum());
            this.numPoints += list.get(i).getNumPoints();
        }
    }
    
    public Point getPartialSum() {
        return partialSum;
    }

    public void setPartialSum(Point partialSum) {
        this.partialSum = partialSum;
    }

    public int getNumPoints() {
        return numPoints;
    }

    public void setNumPoints(int numPoints) {
        this.numPoints = numPoints;
    }

    public Point computeMean(){
        if(numPoints!=0){
            partialSum.scale(numPoints);
        } 
        else{
            for(int i=0; i<partialSum.getFeatures().size(); i++)
                partialSum.getFeatures().set(i, -10.0);
        }
        
        return partialSum;
    }

    public String toString(){
        return "Num Points : " + numPoints + " | " + partialSum;
    }
}