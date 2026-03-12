package com.example;

public class Service {
    private final Client client;

    public Service(Client client) {
        this.client = client;
    }

    public void helper() {
    }

    public void run() {
        helper();
        client.fetch();
    }
}
