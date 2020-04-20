package Arstotzkian_Rail_System;

import java.util.ArrayList;

public class Train {
    private String number;
    private int length;
    private ArrayList<Wagon> carried_wagons = new ArrayList<>();
    private int carried_weight;

    //Do I even need this?
    public Train(String number,int length,ArrayList<Wagon> carried_wagons,int carried_weight){
        this.number = number;
        this.length = length;
        this.carried_wagons = carried_wagons;
        this.carried_weight = carried_weight;
    }

    public Train(String number){
        this.number = number;
        this.length = 0;
        this.carried_wagons = new ArrayList<Wagon>();
    }

    public String getNumber() {
        return number;
    }

    public int getLength() {
        return length;
    }

    public ArrayList<Wagon> getCarried_wagons() {
        return carried_wagons;
    }

    public void addWagon(Wagon wagon){
        this.length++;
        this.carried_wagons.add(wagon);
        this.carried_weight += wagon.getBrut_weight_kg();
    }

    public Wagon takeWagonByWagon(Wagon wagon){
        this.carried_wagons.remove(wagon);
        this.length--;
        this.carried_weight -= wagon.getBrut_weight_kg();
        return wagon;
    }

    //I hope you had checked if wagon with this number exists, before calling this method, if not - it's bad
    public Wagon takeWagonByNumber(String number){
        for(Wagon w:carried_wagons){
            if (w.getNumber() == number){
                Wagon saved_w = w;
                this.length--;
                this.carried_weight -= w.getBrut_weight_kg();
                carried_wagons.remove(w);
                return saved_w;
            }
        }
        return new Wagon();
    }

    public int getCarried_weight() {
        return carried_weight;
    }

    //I don't wanna make it possible to add/subtract weight to/from train

    private void followThisGodDamnTrain(){
        System.out.println("FOLLOW THIS GODDAMN TRAIN CJ!!!");
    }

}
