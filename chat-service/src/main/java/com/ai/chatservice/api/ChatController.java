package com.ai.chatservice.api;

import com.ai.chatservice.core.ChatOrchestrator;
import jakarta.validation.constraints.NotBlank;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * REST API đơn giản:
 *  - POST /api/chat/send  : gửi prompt, nhận câu trả lời
 */
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatOrchestrator orchestrator;
    public ChatController(ChatOrchestrator orchestrator){ this.orchestrator = orchestrator; }

    public record ChatRequest(@NotBlank String prompt, Integer maxNew, Double topP, Double temperature){}
    public record ChatResponse(String reply, String mode, long ms){}

    @PostMapping("/send")
    public ResponseEntity<ChatResponse> send(@RequestBody ChatRequest req){
        long t0 = System.currentTimeMillis();
        var result = orchestrator.answer(
                req.prompt(),
                req.maxNew()==null? 40 : req.maxNew(),
                req.topP()==null? 0.9 : req.topP(),
                req.temperature()==null? 0.8 : req.temperature()
        );
        long ms = System.currentTimeMillis()-t0;
        return ResponseEntity.ok(new ChatResponse(result.text(), result.mode(), ms));
    }
}
