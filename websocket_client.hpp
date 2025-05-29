#pragma once
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <functional>
#include <string>

typedef websocketpp::client<websocketpp::config::asio_client> ws_client;

class WebSocketClient {
public:
    WebSocketClient();
    void connect(const std::string& uri);
    void send(const std::string& message);
    void close();

private:
    void on_message(websocketpp::connection_hdl, ws_client::message_ptr msg);
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_fail(websocketpp::connection_hdl hdl);

    ws_client client;
    websocketpp::connection_hdl connection_hdl;
    bool connected;
    std::thread ws_thread;
};
