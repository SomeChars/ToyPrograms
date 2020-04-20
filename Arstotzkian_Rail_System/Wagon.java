package Arstotzkian_Rail_System;

import java.util.HashMap;
import java.util.Map;

public class Wagon {
    private String number;
    private String category;
    private int wagon_weight_kg;
    private int max_carried_weight;
//    private HashMap<String,Integer> cargo;
    private int net_weight_kg;
    private int brut_weight_kg;
    private Station home_station;
    private Path path;

    public Wagon(String number,String category,int wagon_weight_kg,int max_carried_weight,int net_weight_kg,int brut_weight_kg,Station home_station,Path path){
        this.number = number;
        this.category = category;
        this.wagon_weight_kg = wagon_weight_kg;
        this.max_carried_weight = max_carried_weight;
//        this.cargo = cargo;
        this.net_weight_kg = net_weight_kg;
        this.brut_weight_kg = brut_weight_kg;
        this.home_station = home_station;
        this.path = path;

    }
    public Wagon(String number,String category,int wagon_weight_kg,int max_carried_weight,Station home_station){
        this.number = number;
        this.category = category;
        this.wagon_weight_kg = wagon_weight_kg;
        this.max_carried_weight = max_carried_weight;
        this.home_station = home_station;
        this.net_weight_kg = 0;
        this.brut_weight_kg = wagon_weight_kg;
    }

    public Wagon(String number,String category,int wagon_weight_kg,int max_carried_weight,int net_weight_kg,int brut_weight_kg,Station home_station){
        this.number = number;
        this.category = category;
        this.wagon_weight_kg = wagon_weight_kg;
        this.max_carried_weight = max_carried_weight;
//        this.cargo = cargo;
        this.net_weight_kg = net_weight_kg;
        this.brut_weight_kg = brut_weight_kg;
        this.home_station = home_station;
    }

    //this makes default Wagon, i need it
    public Wagon(){}

    public String getNumber() {
        return number;
    }

    public String getCategory() {
        return category;
    }

    public int getWagon_weight_kg() {
        return wagon_weight_kg;
    }

    public int getMax_carried_weight() {
        return max_carried_weight;
    }

//    public HashMap<String, Integer> getCargo() {
//        return cargo;
//    }

    public int getNet_weight_kg() {
        return net_weight_kg;
    }

    public int getBrut_weight_kg() {
        return brut_weight_kg;
    }

    public Station getHome_station() {
        return home_station;
    }

    public Path getPath() {
        return path;
    }

    public void setNet_weight_kg(int net_weight_kg) {
        this.net_weight_kg = net_weight_kg;
    }

    public void setBrut_weight_kg(int brut_weight_kg) {
        this.brut_weight_kg = brut_weight_kg;
    }

//    public void setCargo(HashMap<String, Integer> cargo) {
//        this.cargo = cargo;
//    }


    public void setPath(Path path) {
        this.path = path;
    }

//    public void addCargo(HashMap<String, Integer> new_cargo){
//        for(Map.Entry c : new_cargo.entrySet()){
//            if (this.cargo.containsKey((String) c.getKey())){
//                this.cargo.replace((String) c.getKey(),this.cargo.get(c.getKey())+(Integer) c.getValue());
//            }
//            else this.cargo.put((String) c.getKey(),(Integer) c.getValue());
//        }
//    }
//
//    public HashMap<String,Integer> unloadCargo(){
//        HashMap<String,Integer> temp = this.cargo;
//        this.cargo = new HashMap<>();
//        return temp;
//    }

    public void addWeight(int weight){
        this.net_weight_kg += weight;
        this.brut_weight_kg += weight;
    }

    public boolean subtractWeight(int weight){
        if (this.brut_weight_kg >= weight){
            this.brut_weight_kg -= weight;
            this.net_weight_kg -= weight;
            return true;
        }
        else return false;
    }
}
