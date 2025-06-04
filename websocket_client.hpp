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
    bool isConnected() const;

private:
    void on_message(websocketpp::connection_hdl, ws_client::message_ptr msg);
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_fail(websocketpp::connection_hdl hdl);
    void reconnect();  // 🔁 新增自動重連功能

    ws_client client;
    websocketpp::connection_hdl connection_hdl;
    bool connected;
    std::string server_uri;  // 🔁 記錄連線 URI

    std::thread ws_thread;
};
