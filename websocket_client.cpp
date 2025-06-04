#include "websocket_client.hpp"
#include <iostream>
#include <thread>
#include <chrono>


WebSocketClient::WebSocketClient() : connected(false) {
    client.init_asio();
    client.set_message_handler(std::bind(&WebSocketClient::on_message, this, std::placeholders::_1, std::placeholders::_2));
    client.set_open_handler(std::bind(&WebSocketClient::on_open, this, std::placeholders::_1));
    client.set_close_handler(std::bind(&WebSocketClient::on_close, this, std::placeholders::_1));
    client.set_fail_handler(std::bind(&WebSocketClient::on_fail, this, std::placeholders::_1));
}

void WebSocketClient::connect(const std::string& uri) {
    server_uri = uri;  // 記住 URI
    websocketpp::lib::error_code ec;
    auto con = client.get_connection(uri, ec);
    if (ec) {
        std::cerr << "❌ WebSocket 連線錯誤: " << ec.message() << std::endl;
        return;
    }
    client.connect(con);
    if (!ws_thread.joinable()) {
        ws_thread = std::thread([this]() {
            client.run();
        });
    }
}

void WebSocketClient::send(const std::string& message) {
    if (connected) {
        websocketpp::lib::error_code ec;
        client.send(connection_hdl, message, websocketpp::frame::opcode::text, ec);
        if (ec) {
            std::cerr << "❌ 傳送訊息錯誤: " << ec.message() << std::endl;
        }
    } else {
        std::cerr << "⚠️ WebSocket 尚未連線，訊息未送出\n";
    }
}

void WebSocketClient::close() {
    if (connected) {
        websocketpp::lib::error_code ec;
        client.close(connection_hdl, websocketpp::close::status::normal, "Closing", ec);
        if (ec) {
            std::cerr << "❌ 關閉錯誤: " << ec.message() << std::endl;
        }
    }
    if (ws_thread.joinable()) {
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
    reconnect();

}

void WebSocketClient::on_fail(websocketpp::connection_hdl hdl) {
    std::cout << "❌ WebSocket 連線失敗" << std::endl;
    connected = false;
    reconnect();

}

bool WebSocketClient::isConnected() const {
    return connected;
}

void WebSocketClient::reconnect() {
    std::this_thread::sleep_for(std::chrono::seconds(3));  // 等三秒再重連
    std::cout << "🔁 嘗試重新連線到 WebSocket...\n";
    connect(server_uri);
}