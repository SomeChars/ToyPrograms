package Arstotzkian_Rail_System;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Station_Manager {
    private Station station;
    private ArrayList<Train> trainsArrived = new ArrayList<>();
    private HashMap<String,Integer> cargoByCategory = new HashMap<>();
    private int emptyWagons = 0;
    private int fullWagons = 0;

//    private static ARS_Support ars_support;

    public Station_Manager(Station station){
        this.station = station;
        cargoByCategory.put("Category1",0);
        cargoByCategory.put("Category2",0);
        cargoByCategory.put("Category3",0);
        this.station.resetFormingTrains();
//        this.ars_support = ars_support;
    }

    public void sortArrivedTrain(){
        if (!station.isAtow()){
           return;
        }
        station.noAtow();
        this.trainsArrived.add(station.getArrived_train());
        for(int i = station.getArrived_train().getLength();i>0;i--){
            Wagon wagon_taken = this.station.getLastWagonFromArrivedTrain(true);
            if (wagon_taken.getNet_weight_kg() > 0) fullWagons++;
            else emptyWagons++;
            if(wagon_taken.getPath().getFinish() == this.station){
                cargoByCategory.replace(wagon_taken.getCategory(),cargoByCategory.get(wagon_taken.getCategory()) + wagon_taken.getNet_weight_kg());
                wagon_taken.subtractWeight(wagon_taken.getNet_weight_kg());
                wagon_taken.setPath(ARS_Support.findPathBetween(this.station,wagon_taken.getHome_station()));
                this.station.addWagonToTrain(wagon_taken,wagon_taken.getPath().getNext(station));
                continue;
            }
            if(wagon_taken.getHome_station() == station){
                this.station.addToBuffer(wagon_taken);
                continue;
            }
            this.station.addWagonToTrain(wagon_taken,wagon_taken.getPath().getNext(station));
        }
    }

    public void releaseBuffer(){
        int bf = station.getBufferSize();
        for(int i = 0;i<bf;i++){
            Wagon wagon = station.getFirstFromBuffer();
            wagon.addWeight((int) (Math.random()*wagon.getMax_carried_weight()/2 + wagon.getMax_carried_weight()/2));
            wagon.setPath(ARS_Support.findPathBetween(station,ARS_Support.chooseRandomStation(station)));
            station.addWagonToTrain(wagon,wagon.getPath().getNext(station));
        }
    }

    public void makeNewRandomWagon(){
        int category = (int) (Math.random()*3) + 1;
        int number = (int) (Math.random()*10000);
        station.addToBuffer(new Wagon(station.getName()+"_"+number+"_C"+category,"Category"+category,1000,10000+10000*category,station));
    }

    public void makeNewWagon(String number,String category,int wagon_weight_kg,int max_carried_weight,Station home_station){
        station.addToBuffer(new Wagon(number,category,wagon_weight_kg,max_carried_weight,home_station));
    }

    //your ad could be here

    public void addSomeNewRandomWagons(int count){
        for (int i = 0;i<count;i++){
            makeNewRandomWagon();
        }
    }

    public ArrayList<Station> getConnections(){
        return station.getConnectedStations();
    }

    public boolean checkStation(Station station){
        return station == this.station;
    }

    public void arrivedTrainsList(){
        if (trainsArrived.size() == 0) return;
        System.out.println(station.getName() + " arrived trains");
        for (Train t: trainsArrived){
            System.out.println(t.getNumber()+"   "+t.getLength()+" wagons   "+t.getCarried_weight()+" kg of cargo");
        }
        System.out.println("--------------");
    }

    public void formingTrainsList() {
        System.out.println(station.getName() + " forming trains");
        for (Station s: station.getConnectedStations()){
            Train t = station.getTrainsOnWays().get(s);
            System.out.println(t.getNumber()+"   "+t.getLength()+" wagons   "+t.getCarried_weight()+" kg of cargo   Forming to station "+s.getName());
        }
        System.out.println("--------------");
    }

    public void cargoList(){
        System.out.println(station.getName()+" cargo by categories");
        int all_cargo = 0;
        for (int i = 1;i <= cargoByCategory.size();i++){
            System.out.println("Category"+i+" cargo "+cargoByCategory.get("Category"+i)+" kg");
            all_cargo += cargoByCategory.get("Category"+i);
        }
        System.out.println("All cargo "+all_cargo+" kg");
        System.out.println("--------------");
    }

    public void importantList(){
        System.out.println(station.getName()+" useful list");
        System.out.print("Arrived trains "+trainsArrived.size()+"   Formed trains "+station.getSentTrains()+"   full/empty wagons "+fullWagons+"/"+emptyWagons);
        if (emptyWagons > 0) System.out.println(" that's "+fullWagons/emptyWagons);
        else System.out.println();
        System.out.println("--------------");

    }

    public int trainsArrived(){
        return trainsArrived.size();
    }

    public void formedTrainInfoTxt() throws IOException {
        File f = new File(station.getName()+"_formed_trains.txt");
        FileWriter fw = new FileWriter(f);
        fw.write("--------------\n");
        for(String s:station.getFormedTrainInfo()){
            fw.write(s+"\n");
            fw.write("--------------\n");
        }
        fw.close();
    }
}
