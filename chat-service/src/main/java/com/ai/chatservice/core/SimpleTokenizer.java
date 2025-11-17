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
        Integer id = wordToId.get(w);
        if (id != null) return id;
        int newId = idToWord.size();
        wordToId.put(w, newId);
        idToWord.add(w);
        return newId;
    }

    /**
     * Đổi câu chữ thành danh sách token (mỗi token là 1 số).
     */
    public int[] encode(String text) {
        text = text.toLowerCase()
                .replaceAll("[,.?!;:]", ""); // bỏ dấu câu đơn giản

        String[] words = text.split("\\s+");
        List<Integer> ids = new ArrayList<>();

        for (String raw : words) {
            String w = raw.trim();
            if (w.isEmpty()) continue;
            ids.add(addWord(w));
        }

        int[] arr = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            arr[i] = ids.get(i);
        }
        return arr;
    }

    /**
     * Đổi danh sách token thành câu chữ.
     */
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= idToWord.size()) continue;
            sb.append(idToWord.get(id)).append(" ");
        }
        return sb.toString().trim();
    }

    // Thêm hàm normalize để dùng chung cho map hỏi–đáp
    public String normalize(String text) {
        return text.toLowerCase()
                .replaceAll("[,.?!;:]", "")
                .replaceAll("\\s+", " ")
                .trim();
    }
}
