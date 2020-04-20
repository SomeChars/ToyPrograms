package Arstotzkian_Rail_System;

public class Way {
    private Station start;
    private Station destination;
    public Way(Station start,Station destination){
        this.start = start;
        this.destination = destination;
    }

    public Station getStart() {
        return start;
    }

    public Station getDestination() {
        return destination;
    }
}
