package com.ai.chatservice.api;

import com.ai.chatservice.service.ChatService;
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

    public ChatController(ChatService chat) {
        this.chat = chat;
    }

    @PostMapping("/send")
    public String send(@RequestBody ChatRequest req) {
        return chat.chat(req.message);
    }

    public record ChatRequest(String message) {}
}
