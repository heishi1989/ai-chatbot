package com.ai.chatservice.service;

import com.ai.chatservice.core.MarkovChatEngine;
import com.ai.chatservice.core.SimpleTokenizer;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Service xử lý logic chat:
 * - Load dữ liệu mẫu khi khởi động
 * - Gọi MarkovChatEngine để sinh câu trả lời
 */
@Service
public class ChatService {

    private final SimpleTokenizer tokenizer = new SimpleTokenizer();
    private final MarkovChatEngine engine = new MarkovChatEngine();
    private boolean trained = false;

    public ChatService() {
        try {
            // Đọc file data huấn luyện (do bạn tự tạo)
            String data = Files.readString(Path.of("data/training.txt"));
            engine.train(tokenizer, data);
            trained = true;
            System.out.println(">>> ChatService: trained from data/training.txt");
        } catch (Exception e) {
            System.err.println("WARNING: Cannot load training data: " + e.getMessage());
        }
    }

    /**
     * Tạo câu trả lời từ prompt người dùng.
     */
    public String chat(String prompt) {
        System.out.println("REQ = " + prompt);

        if (!trained) {
            return "Hi, mình là bot demo, hiện chưa được huấn luyện dữ liệu nên chưa trả lời hay được. Bạn hãy thêm file data/training.txt nhé!";
        }

        String answer = engine.generate(tokenizer, prompt, 20);
        if (answer == null || answer.isBlank()) {
            return "Mình chưa nghĩ ra câu trả lời phù hợp từ dữ liệu đã học, bạn thử hỏi lại cách khác nha.";
        }

        return answer;
    }
}
