package Arstotzkian_Rail_System;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Station {
    private String name;
    private ArrayList<Station> connectedStations = new ArrayList<>();
    private int max_train_length;
    private int max_train_weight;
    private HashMap<Station,Train> trains_on_ways = new HashMap<>();
    private int way_number;
    private Train arrived_train;
    private ArrayList<Wagon> station_wagon_buffer = new ArrayList<>();
    private int sentTrains = 0;
    private boolean atow = false;
    private ArrayList<String> formedTrainInfo = new ArrayList<>();

    public Station(String name,int max_train_length,int max_train_weight){
        this.name = name;
        this.max_train_length = max_train_length;
        this.max_train_weight = max_train_weight;
    }

    public Station(String name,int max_train_length,int max_train_weight,int way_number){
        this.name = name;
        this.max_train_length = max_train_length;
        this.max_train_weight = max_train_weight;
        this.way_number = way_number;
    }

    public String getName(){
        return this.name;
    }

    public Train getArrived_train(){
        return this.arrived_train;
    }

    public ArrayList<Station> getConnectedStations(){
        return connectedStations;
    }

    public void addConnection(Station station){
        this.connectedStations.add(station);
        this.way_number++;
        this.trains_on_ways.put(station,new Train("Train_from_"+this.name+"_to_"+station.getName()+" "+(int) (Math.random()*10000)));
    }

    public void receiveTrain(Train train){
        this.arrived_train = train;
        atow = true;
    }

    public void sendTrain(Station station){
        Train t = trains_on_ways.get(station);
        formedTrainInfo.add(t.getNumber()+"   length "+t.getLength()+"   carried weight "+t.getCarried_weight());
        sentTrains++;
        station.receiveTrain(t);
        this.trains_on_ways.replace(station,new Train("Train_from_"+this.name+"_to_"+station.getName()+" "+(int) (Math.random()*10000)));
    }

    public Wagon getLastWagonFromArrivedTrain(boolean remove){
        Wagon saved_wagon = this.arrived_train.getCarried_wagons().get(this.arrived_train.getLength() - 1);
        if (remove){
            return this.arrived_train.takeWagonByWagon(saved_wagon);
        }
        return saved_wagon;
    }

    public void addWagonToTrain(Wagon wagon,Station station){
        this.trains_on_ways.get(station).addWagon(wagon);
        if (this.trains_on_ways.get(station).getCarried_weight() >= this.max_train_weight || this.trains_on_ways.get(station).getLength() == this.max_train_length){
            this.sendTrain(station);
        }
    }

    public void addToBuffer(Wagon wagon){
        this.station_wagon_buffer.add(wagon);
    }

    public Wagon getFirstFromBuffer(){
        return this.station_wagon_buffer.remove(0);
    }

    public int getBufferSize(){
        return this.station_wagon_buffer.size();
    }

    public void setWay_number(int way_number){
        this.way_number = way_number;
    }

    public HashMap<Station,Train> getTrainsOnWays(){
        return trains_on_ways;
    }

    public int getSentTrains() {
        return sentTrains;
    }

    public boolean isAtow(){
        return atow;
    }

    public void noAtow(){
        atow = false;
    }

    public void resetFormingTrains(){
        for (Station s: trains_on_ways.keySet()) {
            trains_on_ways.replace(s,new Train("Train_from_"+this.name+"_to_"+s.getName()+" "+(int) (Math.random()*10000)));
        }
    }

    public ArrayList<String> getFormedTrainInfo(){
        return formedTrainInfo;
    }

}

