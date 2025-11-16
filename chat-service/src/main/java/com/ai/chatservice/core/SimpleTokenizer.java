package com.ai.chatservice.core;

import java.util.*;

/**
 * Bộ tách từ cực kỳ đơn giản:
 * - Chỉ chia câu theo khoảng trắng và ký tự đặc biệt.
 * - Mỗi từ được đổi thành 1 số (token).
 *
 * Ở mức độ demo, tokenizer này là đủ đơn giản.
 * Trong mô hình AI thật (GPT), tokenizer phức tạp hơn rất nhiều (BPE).
 */
public class SimpleTokenizer {

    private final Map<String, Integer> wordToId = new HashMap<>();
    private final List<String> idToWord = new ArrayList<>();

    public SimpleTokenizer() {
        // Thêm token đặc biệt (nếu cần)
        addWord("<unk>");
    }

    private int addWord(String w) {
        if (!wordToId.containsKey(w)) {
            int id = idToWord.size();
            wordToId.put(w, id);
            idToWord.add(w);
            return id;
        }
        return wordToId.get(w);
    }

    /**
     * Đổi câu chữ thành danh sách token (mỗi token là 1 số).
     */
    public int[] encode(String text) {
        text = text.toLowerCase()
                .replaceAll("[,.?!;:]", ""); // bỏ dấu câu

        String[] words = text.split("\\s+");

        int[] ids = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            String w = words[i].trim();
            if (w.isEmpty()) continue;
            ids[i] = addWord(w);
        }
        return ids;
    }

    /**
     * Đổi danh sách token thành câu chữ.
     */
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id < idToWord.size()) {
                sb.append(idToWord.get(id)).append(" ");
            }
        }
        return sb.toString().trim();
    }
}
