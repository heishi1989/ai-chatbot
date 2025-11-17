package com.ai.chatservice.service;

import com.ai.chatservice.core.*;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.nio.file.Files;

/**
 * Service xử lý logic chat:
 * - Load conversations.jsonl khi khởi động
 * - Nếu câu hỏi trùng với dữ liệu → trả lại đúng câu trả lời đã học
 * - Ngược lại → dùng MarkovChatEngine để sinh câu bắt chước style
 */
@Service
public class ChatService {

    private final SimpleTokenizer tokenizer = new SimpleTokenizer();
    private final MarkovChatEngine markov = new MarkovChatEngine();
    private final ConversationMemory memory = new ConversationMemory(tokenizer);

    private boolean trained = false;

    public ChatService() {
        try {
            String file = "data/conversations.jsonl";

            if (!Files.exists(Path.of(file))) {
                System.err.println("WARNING: conversations.jsonl not found, fallback Markov only.");
                return;
            }

            // 1) Load cặp hỏi–đáp
            memory.loadFromJsonl(file);

            // 2) Lấy toàn bộ assistant content để train Markov
            String assistantCorpus = memory.buildAssistantCorpus();
            markov.train(tokenizer, assistantCorpus);

            trained = true;
            System.out.println(">>> ChatService: trained from " + file);

        } catch (Exception e) {
            System.err.println("WARNING: Cannot load training data: " + e.getMessage());
        }
    }

    public String chat(String prompt) {
        System.out.println("REQ = " + prompt);

        if (!trained) {
            return "Hi, mình là bot demo, hiện chưa được huấn luyện dữ liệu nên chưa trả lời hay được. Bạn hãy thêm file data/conversations.jsonl nhé!";
        }

        // 1) Thử tìm câu trả lời trực tiếp từ conversation memory
        String direct = memory.findDirectReply(prompt);
        if (direct != null && !direct.isBlank()) {
            return direct;
        }

        // 2) Nếu không có match -> dùng Markov để bắt chước style
        String answer = markov.generate(tokenizer, prompt, 30);
        if (answer == null || answer.isBlank()) {
            return "Mình chưa nghĩ ra câu trả lời phù hợp từ dữ liệu đã học, bạn thử hỏi lại cách khác nha.";
        }

        return answer;
    }
}
