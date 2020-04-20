package Arstotzkian_Rail_System;

import java.util.ArrayList;

public class Path {
    private ArrayList<Station> path = new ArrayList<>();
    private Station start;
    private Station finish;

    public Path(Station start){
        this.path = new ArrayList<>();
        this.start = start;
        this.finish = start;
        this.path.add(start);
    }

    public Path(Station start,ArrayList<Station> path){
        this.path = path;
        this.start = start;
        this.finish = path.get(path.size()-1);
    }

    public void addStation(Station station){
        this.path.add(station);
        this.finish = station;
    }

    public Station getStart(){
        return this.start;
    }

    public Station getFinish(){
        return this.finish;
    }

    public Station getNext(Station station){
        return this.path.get(this.path.indexOf(station) + 1);
    }

    public ArrayList<Station> getPath() {
        return path;
    }
}
