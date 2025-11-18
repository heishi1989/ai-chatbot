package com.ai.chatservice.core;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Ghi nhớ các cặp (user -> assistant) từ file training.
 *
 * - Đọc file JSONL do bạn cung cấp (mỗi dòng là 1 object {"messages":[...]}).
 * - Với mỗi block:
 *      user: "..."  ->  assistant: "..."
 *   sẽ lưu vào map để trả lời trực tiếp khi người dùng hỏi giống y chang.
 * - Đồng thời gọi MarkovChatEngine.trainPair(...) để engine Markov học luôn
 *   cả context user + assistant.
 */
public class ConversationMemory {

    private final SimpleTokenizer tokenizer;

    // key: câu user đã chuẩn hoá (normalize)
    // value: danh sách các câu trả lời assistant có thể dùng
    private final Map<String, List<String>> userToReplies = new HashMap<>();

    private final Random rnd = new Random(123);

    public ConversationMemory(SimpleTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    /**
     * Đọc file training dạng nhiều khối JSON:
     *
     *  {"messages":[
     *    {"role":"system","content":"..."},
     *    {"role":"user","content":"..."},
     *    {"role":"assistant","content":"..."}
     *  ]}
     *  {"messages":[ ... ]}
     *  ...
     *
     * @param filePath đường dẫn file (vd: "data/training.jsonl")
     * @param engine   MarkovChatEngine để train thêm (có thể null nếu không dùng)
     */
    public void load(String filePath, MarkovChatEngine engine) throws IOException {

        // Đọc hết file thành 1 chuỗi (vì mỗi block có thể nhiều dòng)
        String all = Files.readString(Path.of(filePath), StandardCharsets.UTF_8);

        // 1) Tìm từng khối {"messages":[ ... ]}
        Pattern convPat = Pattern.compile(
                "\\{\\\"messages\\\":\\[(.+?)]}",
                Pattern.DOTALL
        );
        Matcher mConv = convPat.matcher(all);

        int convCount = 0;

        while (mConv.find()) {
            convCount++;
            String block = mConv.group(1); // phần bên trong [... ] của messages

            // 2) Trong mỗi block, tìm từng message:
            //    {"role":"xxx","content":"yyy"}
            Pattern msgPat = Pattern.compile(
                    "\\{\\\"role\\\":\\\"(system|user|assistant)\\\",\\\"content\\\":\\\"(.*?)\\\"}",
                    Pattern.DOTALL
            );
            Matcher mMsg = msgPat.matcher(block);

            String lastUser = null;

            while (mMsg.find()) {
                String role = mMsg.group(1);      // system / user / assistant
                String content = unescape(mMsg.group(2)).trim(); // text thực tế

                if ("user".equals(role)) {
                    // gặp user -> ghi nhớ, chờ assistant phía sau
                    lastUser = content;

                } else if ("assistant".equals(role) && lastUser != null) {
                    // khi gặp assistant và đã có user ngay trước đó
                    // => tạo cặp (user -> assistant)

                    // 1) lưu để trả lời trực tiếp
                    String key = tokenizer.normalize(lastUser);
                    userToReplies
                            .computeIfAbsent(key, k -> new ArrayList<>())
                            .add(content);

                    // 2) cho Markov học luôn context user+assistant
                    if (engine != null) {
                        engine.trainPair(tokenizer, lastUser, content);
                    }

                    // nếu mỗi block chỉ 1 cặp user/assistant thì reset
                    lastUser = null;
                }
                // role = system thì bỏ qua
            }
        }

        System.out.println("ConversationMemory loaded " + convCount + " conversations.");
        System.out.println("Distinct user patterns: " + userToReplies.size());
    }

    /**
     * Giải mã các escape đơn giản từ JSON:
     *   \\n  -> xuống dòng
     *   \\\" -> "
     * (đủ dùng cho file training do chính mình tạo)
     */
    private String unescape(String s) {
        return s
                .replace("\\n", "\n")
                .replace("\\\"", "\"");
    }

    /**
     * Tìm câu trả lời "chuẩn" nếu user hỏi giống hệt như trong training.
     * - Nếu tìm thấy nhiều câu trả lời cho 1 câu hỏi, sẽ chọn ngẫu nhiên 1 câu.
     */
    public String findDirectReply(String userInput) {
        String key = tokenizer.normalize(userInput);
        List<String> replies = userToReplies.get(key);
        if (replies == null || replies.isEmpty()) return null;
        return replies.get(rnd.nextInt(replies.size()));
    }
}
