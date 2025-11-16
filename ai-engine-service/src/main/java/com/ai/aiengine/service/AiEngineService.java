package com.ai.aiengine.service;

import com.ai.aiengine.core.MarkovChatEngine;
import com.ai.aiengine.core.SimpleTokenizer;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;

@Service
public class AiEngineService {

    private final SimpleTokenizer tokenizer = new SimpleTokenizer();
    private final MarkovChatEngine engine = new MarkovChatEngine();
    private boolean trained = false;

    public AiEngineService() {
        try {
            // Đọc file training giống ChatService
            String data = Files.readString(Path.of("data/training.txt"));
            engine.train(tokenizer, data);
            trained = true;
            System.out.println(">>> AiEngineService: trained from data/training.txt");
        } catch (Exception e) {
            System.err.println("WARNING: Cannot load training data in AiEngineService: " + e.getMessage());
        }
    }

    public String generateReply(String input) {
        System.out.println("AI-ENGINE REQ = " + input);

        if (!trained) {
            return "Hi, mình là AI Engine, hiện chưa được huấn luyện dữ liệu (data/training.txt).";
        }

        String answer = engine.generate(tokenizer, input, 20);

        if (answer == null || answer.isBlank()) {
            return "Mình chưa nghĩ ra câu trả lời phù hợp, bạn thử hỏi cách khác nhé.";
        }

        return answer;
    }
}
