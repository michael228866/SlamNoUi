#include "websocket_client.hpp"
#include <iostream>
#include <thread>

WebSocketClient::WebSocketClient() : connected(false) {
    client.init_asio();
    client.set_message_handler(std::bind(&WebSocketClient::on_message, this, std::placeholders::_1, std::placeholders::_2));
    client.set_open_handler(std::bind(&WebSocketClient::on_open, this, std::placeholders::_1));
    client.set_close_handler(std::bind(&WebSocketClient::on_close, this, std::placeholders::_1));
    client.set_fail_handler(std::bind(&WebSocketClient::on_fail, this, std::placeholders::_1));
}

void WebSocketClient::connect(const std::string& uri) {
    websocketpp::lib::error_code ec;
    auto con = client.get_connection(uri, ec);
    if (ec) {
        std::cerr << "❌ WebSocket 連線錯誤: " << ec.message() << std::endl;
        return;
    }
    client.connect(con);
    ws_thread = std::thread([this]() {
        client.run();
    });
}

void WebSocketClient::send(const std::string& message) {
    if (connected) {
        websocketpp::lib::error_code ec;
        client.send(connection_hdl, message, websocketpp::frame::opcode::text, ec);
        if (ec) {
            std::cerr << "❌ 傳送訊息錯誤: " << ec.message() << std::endl;
        }
    }
}

void WebSocketClient::close() {
    if (connected) {
        websocketpp::lib::error_code ec;
        client.close(connection_hdl, websocketpp::close::status::normal, "Closing", ec);
        if (ec) {
            std::cerr << "❌ 關閉錯誤: " << ec.message() << std::endl;
        }
        ws_thread.join();
    }
}

void WebSocketClient::on_message(websocketpp::connection_hdl, ws_client::message_ptr msg) {
    std::cout << "📩 收到訊息：" << msg->get_payload() << std::endl;
}

void WebSocketClient::on_open(websocketpp::connection_hdl hdl) {
    std::cout << "✅ WebSocket 已連線" << std::endl;
    connection_hdl = hdl;
    connected = true;
}

void WebSocketClient::on_close(websocketpp::connection_hdl hdl) {
    std::cout << "❌ WebSocket 關閉" << std::endl;
    connected = false;
}

void WebSocketClient::on_fail(websocketpp::connection_hdl hdl) {
    std::cout << "❌ WebSocket 連線失敗" << std::endl;
    connected = false;
}
