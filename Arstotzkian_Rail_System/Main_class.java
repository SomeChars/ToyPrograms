package Arstotzkian_Rail_System;

import java.io.IOException;
import java.util.ArrayList;

public class Main_class {
    //GLORY TO ARSTOTZKA!!!
    public static void main(String[] args) throws IOException {
        Station station1 = new Station("Station1",10,200000,2);
        Station station2 = new Station("Station2",10,200000,3);
        Station station3 = new Station("Station3",10,200000,2);
        Station station4 = new Station("Station4",10,200000,3);
        Station station5 = new Station("Station5",10,200000,2);
        Station station6 = new Station("Station6",10,200000,3);
        Station station7 = new Station("Station7",10,200000,3);
        ARS_Support.addStation(station1);
        ARS_Support.addStation(station2);
        ARS_Support.addStation(station3);
        ARS_Support.addStation(station4);
        ARS_Support.addStation(station5);
        ARS_Support.addStation(station6);
        ARS_Support.addStation(station7);
        ARS_Support.addWay(station1,station2);
        ARS_Support.addWay(station2,station3);
        ARS_Support.addWay(station3,station4);
        ARS_Support.addWay(station4,station5);
        ARS_Support.addWay(station5,station6);
        ARS_Support.addWay(station6,station7);
        ARS_Support.addWay(station1,station7);
        ARS_Support.addWay(station7,station2);
        ARS_Support.addWay(station6,station4);
        ARS_Support.findAllPaths(true);
        ARS_Support.findPathBetween(station1,station2);
        Station_Manager station_manager1 = new Station_Manager(station1);
        Station_Manager station_manager2 = new Station_Manager(station2);
        Station_Manager station_manager3 = new Station_Manager(station3);
        Station_Manager station_manager4 = new Station_Manager(station4);
        Station_Manager station_manager5 = new Station_Manager(station5);
        Station_Manager station_manager6 = new Station_Manager(station6);
        Station_Manager station_manager7 = new Station_Manager(station7);

        station_manager1.addSomeNewRandomWagons(20);
        station_manager2.addSomeNewRandomWagons(20);
        station_manager3.addSomeNewRandomWagons(20);
        station_manager4.addSomeNewRandomWagons(20);
        station_manager5.addSomeNewRandomWagons(20);
        station_manager6.addSomeNewRandomWagons(20);
        station_manager7.addSomeNewRandomWagons(20);

        station_manager1.releaseBuffer();


        System.out.println("--------------");

        station_manager2.sortArrivedTrain();
        station_manager2.arrivedTrainsList();
        station_manager2.cargoList();
        station_manager2.importantList();

        station_manager7.sortArrivedTrain();
        station_manager7.arrivedTrainsList();
        station_manager7.cargoList();
        station_manager7.importantList();

        station_manager1.formingTrainsList();
        station_manager1.formedTrainInfoTxt();

        System.out.println("\nHello My Glorious Arstozka!");


        //ceasar is just c + zar = czar
    }
}