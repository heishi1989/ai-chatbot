package com.ai.chatservice.api;

import com.ai.chatservice.service.ChatService;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.*;

/**
 * API REST để chat với A.I:
 * - Gửi { "message": "xin chào" }
 * - Nhận về: câu trả lời do A.I sinh ra
 */
@RestController
@RequestMapping("/api/chat")
@CrossOrigin
public class ChatController {

    private final ChatService chat;

    private final KafkaTemplate<String, String> kafkaTemplate;

    private static final String CHAT_REQUEST_TOPIC = "chat-request";

    public ChatController(ChatService chat, KafkaTemplate<String, String> kafkaTemplate) {
        this.chat = chat;
        this.kafkaTemplate = kafkaTemplate;
    }

    @PostMapping("/send")
    public String send(@RequestBody ChatRequest req) {
        return chat.chat(req.message);
    }

    public record ChatRequest(String message) {}

    @PostMapping
    public String sendMessage(@RequestBody String userMessage) {
        // Gửi message lên Kafka
        kafkaTemplate.send(CHAT_REQUEST_TOPIC, userMessage);
        return "Đã gửi câu hỏi tới AI Engine: " + userMessage;
    }
}
