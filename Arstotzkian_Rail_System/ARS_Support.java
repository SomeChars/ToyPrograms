package Arstotzkian_Rail_System;

import java.util.ArrayList;

public class ARS_Support {
    private static ArrayList<Station> stations = new ArrayList<>();
    private static ArrayList<Way> ways = new ArrayList<>();
    private static ArrayList<Path> paths = new ArrayList<>();

//    public ARS_Support(){
//        stations = new ArrayList<>();
//        ways = new ArrayList<>();
//        paths = new ArrayList<>();
//    }
//
//    public ARS_Support(ArrayList<Station> stations){
//        this.stations = stations;
//    }
//
//    public ARS_Support(ArrayList<Station> stations,ArrayList<Way> ways){
//        this.stations = stations;
//        this.ways = ways;
//    }

    public static void addStation(Station station){
        stations.add(station);
    }

    public static void addWay(Way way){
        ways.add(way);
        way.getStart().addConnection(way.getDestination());
        way.getDestination().addConnection(way.getStart());
    }

    public static void addWay(Station station1,Station station2){
        ways.add(new Way(station1,station2));
        station1.addConnection(station2);
        station2.addConnection(station1);
    }

    public static ArrayList<Station> getStations() {
        return stations;
    }

    public static ArrayList<Way> getWays() {
        return ways;
    }

    public static void findAllPaths(boolean removeOld){
        if (removeOld) paths.clear();
        for (Station s: stations){
            paths.addAll(D(s));
        }
    }

    private static ArrayList<Path> D(Station station){
        ArrayList<Path> best_paths = new ArrayList<>();
        ArrayList<Station> reached = new ArrayList<>();
        reached.add(station);
        ArrayList<Path> new1_best_paths = new ArrayList<>();
        new1_best_paths.add(new Path(station));
        ArrayList<Path> new2_best_paths = new ArrayList<>();
        for(int i = 0;i < stations.size();i++){
            if (reached.size() == stations.size()) continue;
            for (Path p: new1_best_paths){
                for (Way w: ways){
                    if (w.getStart() == p.getFinish() && !reached.contains(w.getDestination())){
                        Path p_copy = new Path(p.getStart(),(ArrayList<Station>) p.getPath().clone());
                        p_copy.addStation(w.getDestination());
                        new2_best_paths.add(p_copy);
                        reached.add(w.getDestination());
                    }
                    if (w.getDestination() == p.getFinish() && !reached.contains(w.getStart())){
                        Path p_copy = new Path(p.getStart(),(ArrayList<Station>) p.getPath().clone());
                        p_copy.addStation(w.getStart());
                        new2_best_paths.add(p_copy);
                        reached.add(w.getStart());
                    }
                }
            }
            best_paths.addAll(new1_best_paths);
            new1_best_paths.clear();
            new1_best_paths.addAll(new2_best_paths);
            new2_best_paths.clear();
        }
        best_paths.addAll(new1_best_paths);
        best_paths.remove(0);
        return best_paths;
    }

    public static Path findPathBetween(Station station1,Station station2){
        for (Path p: paths){
            if (p.getStart() == station1 && p.getFinish() == station2) return p;
        }
        return new Path(station1);
    }

    public static Station chooseRandomStation(Station station){
        Station s;
        s = stations.get((int) (Math.random()*stations.size()));
        while (s == station){
            s = stations.get((int) (Math.random()*stations.size()));
        }
        return s;
    }

}
