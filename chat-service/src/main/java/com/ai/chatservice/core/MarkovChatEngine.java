package com.ai.chatservice.core;

import java.util.*;

/**
 * A.I kiểu “Markov” cực dễ hiểu:
 *
 * - A.I đọc toàn bộ văn bản huấn luyện.
 * - Nó đếm xem: Sau từ A → thường xuất hiện từ B nào?
 *
 * Ví dụ:
 *   "xin chào bạn"
 *   "xin lỗi bạn"
 *
 * Sau từ "xin":
 *   {"chào":1, "lỗi":1}
 *
 * ⇒ Khi sinh câu, nếu từ cuối là "xin", A.I sẽ chọn “chào” hoặc “lỗi”.
 *
 * Như vậy:
 * - A.I nói có nghĩa thật (vì học từ dữ liệu thật)
 * - Nhưng chỉ ở mức đơn giản, bắt chước mẫu đã học
 */
public class MarkovChatEngine {

    // Chứa thống kê:
    //   fromToken → (toToken → count)
    private final Map<Integer, Map<Integer, Integer>> transitions = new HashMap<>();

    private final Random random = new Random(42);

    /**
     * Huấn luyện từ 1 đoạn văn bản thô.
     * Dữ liệu càng nhiều → A.I càng nói giống "người" hơn.
     */
    public void train(SimpleTokenizer tokenizer, String text) {

        // Tách câu theo dấu . ! ?
        String[] sentences = text.split("[\\.\\!\\?]+");

        for (String sentence : sentences) {
            sentence = sentence.trim();
            if (sentence.isEmpty()) continue;

            // Chuyển câu thành token
            int[] tokens = tokenizer.encode(sentence);

            // Với mỗi cặp liên tiếp (A → B), tăng đếm
            for (int i = 0; i < tokens.length - 1; i++) {
                int t1 = tokens[i];
                int t2 = tokens[i + 1];

                Map<Integer, Integer> map = transitions.computeIfAbsent(t1, k -> new HashMap<>());
                map.put(t2, map.getOrDefault(t2, 0) + 1);
            }
        }
    }

    /**
     * Sinh câu mới dựa trên prompt.
     * - Lấy từ cuối của prompt làm điểm bắt đầu.
     * - Chọn token tiếp theo theo tần suất.
     */
    public String generate(SimpleTokenizer tokenizer,
                           String prompt,
                           int maxNewTokens) {

        int[] encoded = tokenizer.encode(prompt);
        List<Integer> output = new ArrayList<>();

        int currentToken;

        if (encoded.length == 0) {
            // Nếu prompt rỗng → chọn đại 1 token trong bảng
            currentToken = pickAnyToken();
        } else {
            currentToken = encoded[encoded.length - 1];
        }

        // Sinh tiếp các token mới
        for (int i = 0; i < maxNewTokens; i++) {

            Integer next = pickNext(currentToken);
            if (next == null) break;

            output.add(next);
            currentToken = next;
        }

        return tokenizer.decode(output);
    }

    /**
     * Chọn token bất kỳ làm token bắt đầu.
     */
    private int pickAnyToken() {
        if (transitions.isEmpty()) return 0;
        List<Integer> keys = new ArrayList<>(transitions.keySet());
        return keys.get(random.nextInt(keys.size()));
    }

    /**
     * Chọn token tiếp theo dựa trên tần suất xuất hiện.
     */
    private Integer pickNext(int token) {
        Map<Integer, Integer> freq = transitions.get(token);
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
