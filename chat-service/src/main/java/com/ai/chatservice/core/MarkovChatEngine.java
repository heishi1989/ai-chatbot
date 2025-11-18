package com.ai.chatservice.core;

import java.util.*;

/**
 * Markov n-gram (bậc 1,2,3...)
 * - order = 1  → giống bản cũ
 * - order = 2/3→ nhìn 2/3 token trước để đoán token sau
 */
public class MarkovChatEngine {

    private final int order; // bậc Markov
    // stateKey (ví dụ "12,35") → (nextToken → count)
    private final Map<String, Map<Integer, Integer>> transitions = new HashMap<>();
    private final Random random = new Random(42);

    public MarkovChatEngine(int order) {
        this.order = Math.max(1, order); // không cho nhỏ hơn 1
    }

    /**
     * Huấn luyện từ 1 chuỗi văn bản thô.
     * Có thể gọi nhiều lần với các đoạn text khác nhau.
     */
    public void train(SimpleTokenizer tokenizer, String text) {
        int[] tokens = tokenizer.encode(text);
        if (tokens.length <= order) return;

        // Khởi tạo cửa sổ đầu tiên
        int[] window = new int[order];
        System.arraycopy(tokens, 0, window, 0, order);

        for (int i = order; i < tokens.length; i++) {
            String state = stateKey(window);
            int next = tokens[i];

            Map<Integer, Integer> freq =
                    transitions.computeIfAbsent(state, k -> new HashMap<>());
            freq.put(next, freq.getOrDefault(next, 0) + 1);

            // Trượt cửa sổ: bỏ phần tử đầu, thêm next vào cuối
            System.arraycopy(window, 1, window, 0, order - 1);
            window[order - 1] = next;
        }
    }

    /**
     * Huấn luyện từ 1 cặp (user, assistant).
     * - Ghép user + assistant lại thành 1 chuỗi
     *   để từ của assistant phụ thuộc vào context user.
     */
    public void trainPair(SimpleTokenizer tokenizer,
                          String userText,
                          String assistantText) {
        // Có thể thêm token đặc biệt giữa 2 bên cho rõ ràng
        // VD: "<sep>" nhưng ở đây mình ghép thẳng cho đơn giản
        String combined = userText + " " + assistantText;
        train(tokenizer, combined);
    }

    /**
     * Sinh câu trả lời mới dựa trên prompt.
     */
    public String generate(SimpleTokenizer tokenizer,
                           String prompt,
                           int maxNewTokens) {

        int[] encoded = tokenizer.encode(prompt);
        List<Integer> output = new ArrayList<>();

        int[] window = new int[order];

        if (encoded.length >= order) {
            // Dùng đúng order token cuối của câu hỏi
            System.arraycopy(
                    encoded,
                    encoded.length - order,
                    window,
                    0,
                    order
            );
        } else if (encoded.length > 0) {
            // Nếu ít token hơn order → lặp lại token cuối cho đủ
            Arrays.fill(window, encoded[encoded.length - 1]);
        } else {
            // Prompt rỗng → chọn state bất kỳ
            String any = pickAnyState();
            if (any == null) return "";
            String[] parts = any.split(",");
            for (int i = 0; i < order; i++) {
                window[i] = Integer.parseInt(parts[i]);
            }
        }

        for (int i = 0; i < maxNewTokens; i++) {
            String state = stateKey(window);
            Integer next = pickNext(state);
            if (next == null) break;

            output.add(next);

            // Trượt cửa sổ
            System.arraycopy(window, 1, window, 0, order - 1);
            window[order - 1] = next;
        }

        return tokenizer.decode(output);
    }

    // ========= Helpers =========

    // Chuyển cửa sổ token thành key chuỗi: "id1,id2,..."
    private String stateKey(int[] window) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < window.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(window[i]);
        }
        return sb.toString();
    }

    // Chọn state bất kỳ
    private String pickAnyState() {
        if (transitions.isEmpty()) return null;
        List<String> keys = new ArrayList<>(transitions.keySet());
        return keys.get(random.nextInt(keys.size()));
    }

    // Chọn token tiếp theo dựa trên tần suất xuất hiện
    private Integer pickNext(String state) {
        Map<Integer, Integer> freq = transitions.get(state);
        if (freq == null || freq.isEmpty()) return null;

        int total = freq.values().stream().mapToInt(i -> i).sum();
        int r = random.nextInt(total);

        int cumulative = 0;
        for (Map.Entry<Integer, Integer> e : freq.entrySet()) {
            cumulative += e.getValue();
            if (r < cumulative) {
                return e.getKey();
            }
        }
        return null;
    }
}
