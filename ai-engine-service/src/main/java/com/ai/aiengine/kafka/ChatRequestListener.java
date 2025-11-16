package com.ai.aiengine.kafka;

import com.ai.aiengine.service.AiEngineService;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
public class ChatRequestListener {

    private final AiEngineService aiEngineService;

    public ChatRequestListener(AiEngineService aiEngineService) {
        this.aiEngineService = aiEngineService;
    }

    @KafkaListener(topics = "chat-request", groupId = "ai-engine-group")
    public void handleChatRequest(String userMessage) {
        System.out.println("Nhận message từ chat-service: " + userMessage);

        String reply = aiEngineService.generateReply(userMessage);

        // Bước đơn giản: chưa trả kết quả về, chỉ log ra
        System.out.println("AI trả lời: " + reply);

        // Sau này muốn gửi lại qua topic khác:
        // kafkaTemplate.send("chat-response", reply);
    }
}