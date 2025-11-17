package com.ai.chatservice.core;

import java.nio.file.*;
import java.util.*;

/**
 * Lưu cặp (user → assistant) đơn giản, đọc từ conversations.jsonl
 */
public class ConversationMemory {

    private final Map<String, List<String>> userToReplies = new HashMap<>();

    private final SimpleTokenizer tokenizer;

    public ConversationMemory(SimpleTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public void loadFromJsonl(String filePath) throws Exception {
        List<String> lines = Files.readAllLines(Path.of(filePath));

        String pendingUser = null;  // nhớ câu user gần nhất

        for (String line : lines) {
            if (line == null || line.isBlank()) continue;

            // Nếu là dòng user → lưu lại
            String u = extractContent(line, "\"role\":\"user\"");
            if (u != null) {
                pendingUser = u;
                continue;
            }

            // Nếu là dòng assistant → ghép với pendingUser (nếu có)
            String a = extractContent(line, "\"role\":\"assistant\"");
            if (a != null && pendingUser != null) {
                String key = tokenizer.normalize(pendingUser);
                userToReplies
                        .computeIfAbsent(key, k -> new ArrayList<>())
                        .add(a.trim());

                // reset để lần sau không ghép nhầm
                pendingUser = null;
            }

            // Dòng system thì mình bỏ qua, không cần xử lý
        }

        System.out.println("ConversationMemory loaded pairs = " + userToReplies.size());
    }

    /**
     * Trả về câu trả lời nếu prompt trùng với một user trong dữ liệu.
     */
    public String findDirectReply(String prompt) {
        String key = tokenizer.normalize(prompt);
        List<String> replies = userToReplies.get(key);
        if (replies == null || replies.isEmpty()) return null;
        // nếu có nhiều reply cho cùng 1 câu hỏi → chọn ngẫu nhiên 1 cái
        return replies.get(new Random().nextInt(replies.size()));
    }

    /**
     * Trích content từ JSONL thô, dựa trên marker "role":...
     * Đây là parser thủ công rất đơn giản, chỉ đủ dùng cho file tự quản lý.
     */
    private String extractContent(String line, String roleMarker) {
        int roleIdx = line.indexOf(roleMarker);
        if (roleIdx < 0) return null;

        int contentIdx = line.indexOf("\"content\":\"", roleIdx);
        if (contentIdx < 0) return null;
        contentIdx += "\"content\":\"".length();

        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        for (int i = contentIdx; i < line.length(); i++) {
            char c = line.charAt(i);
            if (escape) {
                // xử lý 1 số escape cơ bản
                if (c == 'n') sb.append('\n');
                else if (c == '\"') sb.append('\"');
                else sb.append(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '\"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Gom toàn bộ câu trả lời assistant thành 1 text lớn để train Markov.
     */
    public String buildAssistantCorpus() {
        StringBuilder sb = new StringBuilder();
        for (List<String> replies : userToReplies.values()) {
            for (String r : replies) {
                sb.append(r).append(". ");
            }
        }
        return sb.toString();
    }
}